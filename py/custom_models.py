import torch
import torch.nn as nn
from transformers import AutoModel
from segmentation_models_pytorch.decoders.deeplabv3.decoder import DeepLabV3PlusDecoder
from segmentation_models_pytorch.base.heads import SegmentationHead
import math
from transformers import Mask2FormerForUniversalSegmentation

class DINOv3EncoderDeeplabV3PlusDecoder(nn.Module):
    def __init__(
            self,
            num_classes=11,
            in_channels=4,
            backbone_kwargs={},
            decoder_kwargs={},
            head_kwargs={},
            **kwargs):
        super(DINOv3EncoderDeeplabV3PlusDecoder, self).__init__()
        self.backbone = AutoModel.from_pretrained(
            pretrained_model_name_or_path="facebook/dinov3-vitl16-pretrain-sat493m"
        )
        self.hidden_size = self.backbone.config.hidden_size
        self.adapt_patch_embed_to_channels(in_channels)

        self.feature_indicies = [9, 39] # indici dei transformer block da cui estrarre le feature map per il decoder
                                        # 10° blocco per la skip connection richiesata dal decoder DeeplabV3+
                                        # 40° blocco (l'ultimo output del backbone) per l'input al modulo ASPP di DeeplabV3+
        
        self.decoder_channels = 256     # numero di canali delle feature map in input al decoder
        skip_channels = 48              # numero di canali delle feature map per la skip connection (proposto dagli autori deeplabv3+)
        
        # Convoluzione 1x1 per adattare la feature map estratta dal backbone alla dimensione richiesta dal decoder per la skip connection
        self.skip_conv = nn.Conv2d(self.hidden_size, skip_channels, kernel_size=1)

        # Simulaziomne della lista dei canali che un encoder CNN avrebbe prodotto
        # Deeplabv3+ utilizza di default l'elemento a indice [1] per i canali della skip connection (ne ha una sola)
        # l'elemento di indice [-1] (l'ultimo) per i canali in input al modulo ASPP
        encoder_channel_list = [0, 0, skip_channels, self.hidden_size]
        # in pratica l'output dell'encoder passa a ASPP, l'output di ASPP viene upsampled 4x, viene aggiunta
        # la skip connection e poi si passa alla segmentation head

        self.deeplab_decoder = DeepLabV3PlusDecoder(
            encoder_channels=encoder_channel_list,
            encoder_depth=5, # dice al decoder quante feature map aspettarsi (in questo caso 5 simulate in channel_list)
                             # prenderà in autonomia solo gli elementi a indice [2] e [-1], perchè encoder classici (resnet)
                             # hanno encoder depth pari a 5 di solito
            out_channels=self.decoder_channels,
            output_stride=16,
            atrous_rates=[12, 24, 36], # valori standard per output_stride=16
            aspp_separable= False,  # se usare o no depthwise separable convs nel modulo ASPP. riducono drasticamente il numero di parametri
                                    # al costo di una leggera perdita di performance. Inizio con False e poi valuto
            aspp_dropout= decoder_kwargs.get("aspp_dropout", 0.5)      # tunable
        )

        self.segmentation_head = SegmentationHead(
            in_channels=self.decoder_channels,
            out_channels=num_classes,
            kernel_size=1,
            activation=None,
            upsampling=4 # congruente con il decoder DeepLabV3+
        )

    def tokens_to_image(self, tokens):
            """
            Effettua il reshape del token output di un transformer block in una feature map 2D
            """
            B, N, C = tokens.shape # batch size, num tokens, hidden size
            grid_size = int(math.sqrt(N)) # per creare l'immagine [sqrt(N), sqrt(N)]
            if grid_size * grid_size != N:
                raise ValueError(f"Il numero di token ({N}) non è un quadrato perfetto. Impossibile effettuare reshape a feature map 2D")
            return tokens.permute(0, 2, 1).reshape(B, C, grid_size, grid_size)

    def get_spatial_tokens(self, hidden_states):
        """
        restituisce una tupla di hidden_state nei quali vengono ignorati i primi 5 token (class token + 4 register token)
        (VISION TRANSFORMERS NEED REGISTERS, https://arxiv.org/pdf/2309.16588)
        """
        return tuple(state[:, 5:, :] for state in hidden_states)

# ToDo: reference forward
    def forward(self, x):
        # x shape: [B, C, H, W]
        B, C, H, W = x.shape

        # Passaggio nel backbone, ottengo le hidden states (una per ognuno dei 40 transformer block di DINOv3)
        # due di questi token saranno trasformati dal neck in feature map per il decoder
        backbone_outputs = self.backbone(x, output_hidden_states=True)
        hidden_states = backbone_outputs.hidden_states      # tuple di tensori, ognuno ha shape [B, num_tokens, hidden_size]
        # Rimuovo CLS token e Register tokens da tutte le hidden states
        spatial_hidden_states = self.get_spatial_tokens(hidden_states) # [B, num_tokens-5 (1024), hidden_size]

        # Creazione delle feature map dai token
        # Feature map per il modulo ASPP di DeeplabV3+
        aspp_hidden_state = spatial_hidden_states[-1]  # output finale del backbone (output del 40° transformer block di DINOv3)
        aspp_feature_map = self.tokens_to_image(aspp_hidden_state) # [B, hidden_size, H', W']

        # Feature map per la skip connection del decoder DeeplabV3+
        skip_hidden_state = spatial_hidden_states[self.feature_indicies[0]] # output del 10° transformer block di DINOv3
        skip_feature_map = self.tokens_to_image(skip_hidden_state) # [B, hidden_size, H'', W'']

        # Adattamento della skip connection:
        # - upsampling a 1/4 della risoluzione dell'input (richiesto da deeplabv3+)
        # - convoluzione 1x1 per ridurre il numero di canali a skip_channels (richiesto da deeplabv3+)
        skip_feature_map = nn.functional.interpolate(
            skip_feature_map,
            size= (H // 4, W // 4), # 1/4 della risoluzione dell'input
            mode='bilinear', 
            align_corners=False
        )
        # Applica la convoluzione 1x1 alla feature map della skip connection per ridurre i canali
        skip_feature_map = self.skip_conv(skip_feature_map) # [B, skip_channels, H/4, W/4]

        decoder_features = [None, None, skip_feature_map, aspp_feature_map]
        decoder_output = self.deeplab_decoder(*decoder_features) # prende da solo indice [2] e [-1] della list (unpacked)
        logits = self.segmentation_head(decoder_output)
        return logits


    def adapt_patch_embed_to_channels(self, in_channels):
        """
        Adatta il patch embedding layer di DINOv3 per ricevere in input in_channels canali.
        inizializza i pesi dei canali aggiuntivi come la media dei pesi esistenti.

        Args:
            in_channels (int): Numero di canali di input desiderati
        """
        print()
        if in_channels == 3:
            print("in_channels=3, nessuna modifica al patch_embedding layer")
            return  # Nessuna modifica necessaria se 3 canali
        
        # Calcola i nuovi pesi e concatenali a quelli vecchi
        original_weights = self.backbone.embeddings.patch_embeddings.weight.clone().detach() # [4096, 3, 16, 16]
        mean_rgb_weights = original_weights.mean(dim=1, keepdim=True)   # [4096, 1, 16, 16]
        num_extra_channels = in_channels - 3
        extra_channel_weights = mean_rgb_weights.repeat(1, num_extra_channels, 1, 1)  # [4096, num_extra_channels, 16, 16]
        new_weights = torch.cat([original_weights, extra_channel_weights] , dim=1) # [4096, in_channels, 16, 16]

        # Sostituisci il patch_embedding vecchio con uno nuovo che accetti in_channels canali
        new_patch_embedding = nn.Conv2d(in_channels, 4096, kernel_size=(16, 16), stride=(16, 16))

        # Assegna i nuovi pesi al patch embedding layer
        new_patch_embedding.weight = nn.Parameter(new_weights)
        # gestisci i bias term se presenti
        if self.backbone.embeddings.patch_embeddings.bias is not None:
           new_patch_embedding.bias = nn.Parameter(self.backbone.embeddings.patch_embeddings.bias.clone().detach())

        # Sostituisci il vecchio patch embedding con il nuovo
        self.backbone.embeddings.patch_embeddings = new_patch_embedding
        self.backbone.config.num_channels = in_channels
        print(f"Adapted DINOv3 patch embedding to accept {in_channels} input channels.")

class Dinov3EncoderMask2FormerDecoder(nn.Module):
    def __init__(
            self,
            num_classes=11,
            in_channels=4,
            backbone_kwargs={},
            decoder_kwargs={},
            head_kwargs={},
            **kwargs
            ):
        super(Dinov3EncoderMask2FormerDecoder, self).__init__()
        # DINOv3 backbone
        self.backbone = AutoModel.from_pretrained(
            pretrained_model_name_or_path="facebook/dinov3-vitl16-pretrain-sat493m"
        )
        self.adapt_patch_embed_to_channels(in_channels)

        # Mask2Former decoder: ho bisogno solo del pixel decoder e del transformer decoder
        mask2former_reference = Mask2FormerForUniversalSegmentation.from_pretrained(
            pretrained_model_name_or_path="facebook/mask2former-swin-large-cityscapes-semantic"
        )
        self.pixel_decoder = mask2former_reference.model.pixel_level_module.decoder
        self.transformer_decoder = mask2former_reference.model.transformer_module

        # Neck: devo adattare gli output di DINOv3 agli input richiesti dai decoder di Mask2Former
        self.dinov3_hidden_size = self.backbone.config.hidden_size
        self.feature_map_indicies = [9, 19, 29, 39] # transformer blocks da cui estrarre i token da convertire in feature maps
        
        # 1 feature map va in (adapter_1), le altre 3 in (input_projections) del pixel decoder
        # calcolo i canali attesi dal decoder per ogni connessione
        adapter1_in_channels = self.pixel_decoder.adapter_1[0].in_channels # 96
        input_projections_in_channels = [
            proj[0].in_channels for proj in self.pixel_decoder.input_projections
        ] # [768, 384, 192]
        self.expected_in_channels = [adapter1_in_channels] + input_projections_in_channels[::-1] # [96, 192, 384, 768]
        
        # mappa i token [1024] ai 4 layer richiesti [96, 192, 384, 768], con convoluzioni 1x1
        self.projection_layers = nn.ModuleList([
            nn.Conv2d(self.dinov3_hidden_size, out_ch, kernel_size=1) for out_ch in self.expected_in_channels
        ])
# ToDo: testare e debuggare il forward
    def forward(self, x):
        # x shape: [B, C, H, W]
        B, C, H, W = x.shape

        backbone_outputs = self.backbone(x, output_hidden_states=True)
        hidden_states = backbone_outputs.hidden_states # ognuno [B, num_tokens, hidden_size]
        # Rimuovo CLS token e Register tokens da tutte le hidden states
        spatial_hidden_states = self.get_spatial_tokens(hidden_states) # ognuno [B, num_tokens-5 (1024), hidden_size]
        
        dino_feature_maps = []
        for i, block_idx in enumerate(self.feature_map_indicies):
            tokens = spatial_hidden_states[block_idx] # [B, 1024, hidden_size]
            feature_map = self.tokens_to_image(tokens) # [B, hidden_size, H', W'
            projected_map = self.projection_layers[i](feature_map) # [B, expected_in_channels[i], H', W']
            dino_feature_maps.append(projected_map)

        # ToDo: debuggare e controllare gli accessi ai dizionari
        pixel_decoder_output = self.pixel_decoder(dino_feature_maps)
        transformer_decoder_output = self.transformer_decoder(
            pixel_decoder_output["pixel_features"],
            pixel_decoder_output("pixel_mask", None)
        )

        # Logit e upsampling finale.
        class_predictions = transformer_decoder_output["class_queries_logits"]
        mask_queries_logits = transformer_decoder_output["mask_queries_logits"]
        
        logits = torch.einsum("bqc,bqhw->bchw", class_predictions, mask_queries_logits) # non ho compreso qui (???)

        # Upsampling alla dimensione originale dell'input
        logits = nn.functional.interpolate(
            logits,
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )

        return logits
    
    def tokens_to_image(self, tokens, dim):
        """
        Effettua il reshape del token output di un transformer block in una feature map 2D
        """
        B, N, C = tokens.shape # batch size, num tokens, hidden size
        grid_size = int(math.sqrt(N)) # per creare l'immagine [sqrt(N), sqrt(N)]
        if grid_size * grid_size != N:
            raise ValueError(f"Il numero di token ({N}) non è un quadrato perfetto. Impossibile effettuare reshape a feature map 2D")
        feature_map=tokens.permute(0, 2, 1).reshape(B, C, grid_size, grid_size)
        
        return feature_map

    def get_spatial_tokens(self, hidden_states):
        """
        restituisce una tupla di hidden_state nei quali vengono ignorati i primi 5 token (class token + 4 register token)
        (VISION TRANSFORMERS NEED REGISTERS, https://arxiv.org/pdf/2309.16588)
        """
        return tuple(state[:, 5:, :] for state in hidden_states)

    def adapt_patch_embed_to_channels(self, in_channels):
        """
        Adatta il patch embedding layer di DINOv3 per ricevere in input in_channels canali.
        inizializza i pesi dei canali aggiuntivi come la media dei pesi esistenti.

        Args:
            in_channels (int): Numero di canali di input desiderati
        """
        print()
        if in_channels == 3:
            print("in_channels=3, nessuna modifica al patch_embedding layer")
            return  # Nessuna modifica necessaria se 3 canali
        
        # Calcola i nuovi pesi e concatenali a quelli vecchi
        original_weights = self.backbone.embeddings.patch_embeddings.weight.clone().detach() # [4096, 3, 16, 16]
        mean_rgb_weights = original_weights.mean(dim=1, keepdim=True)   # [4096, 1, 16, 16]
        num_extra_channels = in_channels - 3
        extra_channel_weights = mean_rgb_weights.repeat(1, num_extra_channels, 1, 1)  # [4096, num_extra_channels, 16, 16]
        new_weights = torch.cat([original_weights, extra_channel_weights] , dim=1) # [4096, in_channels, 16, 16]

        # Sostituisci il patch_embedding vecchio con uno nuovo che accetti in_channels canali
        new_patch_embedding = nn.Conv2d(in_channels, 4096, kernel_size=(16, 16), stride=(16, 16))

        # Assegna i nuovi pesi al patch embedding layer
        new_patch_embedding.weight = nn.Parameter(new_weights)
        # gestisci i bias term se presenti
        if self.backbone.embeddings.patch_embeddings.bias is not None:
           new_patch_embedding.bias = nn.Parameter(self.backbone.embeddings.patch_embeddings.bias.clone().detach())

        # Sostituisci il vecchio patch embedding con il nuovo
        self.backbone.embeddings.patch_embeddings = new_patch_embedding
        self.backbone.config.num_channels = in_channels
        print(f"Adapted DINOv3 patch embedding to accept {in_channels} input channels.")

if __name__ == "__main__":
    print()