import torch
import torch.nn as nn
from transformers import AutoModel
from segmentation_models_pytorch.decoders.deeplabv3.decoder import DeepLabV3PlusDecoder
from segmentation_models_pytorch.base import SegmentationHead
import math

class DINOv3ForSemanticSegmentation(nn.Module):
    def __init__(self, num_classes=11, in_channels=4):
        super(DINOv3ForSemanticSegmentation, self).__init__()
        self.backbone = AutoModel.from_pretrained(
            pretrained_model_name_or_path="facebook/dinov3-vit7b16-pretrain-sat493m",
            device_map="auto"
        )
        self.hidden_size = self.backbone.config.hidden_size
        self.adapt_patch_embed_to_channels(in_channels)

        self.feature_indicies = [9, 39] # indici dei transformer block da cui estrarre le feature map per il decoder
                                        # 10° blocco per la skip connection richiesata dal decoder DeeplabV3+
                                        # 40° blocco (l'ultimo output del backbone) per l'input al modulo ASPP di DeeplabV3+
        
        self.decoder_channels = 256     # numero di canali delle feature map in input al decoder
        skip_channels = 48              # numero di canali delle feature map per la skip connection
        
        # Convoluzione 1x1 per adattare la feature map estratta dal backbone alla dimensione richiesta dal decoder per la skip connection
        self.skip_conv = nn.Conv2d(self.hidden_size, skip_channels, kernel_size=1)

        # Simulaziomne della lista dei canali che un encoder CNN avrebbe prodotto
        # Deeplabv3+ utilizza di default l'elemento a indice [1] per i canali della skip connection (ne ha una sola)
        # l'elemento di indice [-1] (l'ultimo) per i canali in input al modulo ASPP
        encoder_channel_list = [0, skip_channels, 0, 0, self.hidden_size]
        # in pratica l'output dell'encoder passa a ASPP, l'output di ASPP viene upsampled 4x, viene aggiunta
        # la skip connection e poi si passa alla segmentation head

        deeplab_model = DeepLabV3PlusDecoder(
            encoder_channels=encoder_channel_list,
            out_channels=self.decoder_channels,
            output_stride=16,
            atrous_rates=[12, 24, 36] # valori standard per output_stride=16
        )

        self.segmentation_head = SegmentationHead(
            in_channles=self.decoder_channels,
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

    def forward(self, x):
        # x shape: [B, C, H, W]
        B, C, H, W = x.shape

        # Passaggio nel backbone, ottengo le hidden states (una per ognuno dei 40 transformer block di DINOv3)
        # due di questi token saranno trasformati dal neck in feature map per il decoder
        backbone_outputs = self.backbone(x, output_hidden_states=True)
        hidden_states = backbone_outputs.hidden_states      # tuple di tensori, ognuno ha shape [B, num_tokens, hidden_size]

        # Creazione delle feature map dai token
        # Feature map per il modulo ASPP di DeeplabV3+
        aspp_hidden_state = hidden_states[-1]  # output finale del backbone (output del 40° transformer block di DINOv3)
        aspp_feature_map = self.tokens_to_image(aspp_hidden_state) # [B, hidden_size, H', W']

        # Feature map per la skip connection del decoder DeeplabV3+
        skip_hidden_state = hidden_states[self.feature_indicies[0]] # output del 10° transformer block di DINOv3
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

        decoder_features = [None, skip_feature_map, None, None, aspp_feature_map]
        decoder_output = self.decoder(*decoder_features) # prende da solo indice [1] e [-1] della lista
        logits = self.segmentation_head(decoder_output)
        return logits
    # ToDo: testare!





        feature_map = tokens_to_image(last_hidden_state) # [B, hidden_size, H', W']
        skip_feature_map_1, skip_feature_map_2, skip_feature_map_3, skip_feature_map4 = tokens_to_image(hidden_states)

        # Passaggio nel decoder
        # Attualmente implementata solo la prima skip connection. Il decoder smp di DeepLabV3+ si
        # aspetta una lista di features map in input, una per ogni skip connection
        # bisogna implementarne 4 e scegliere gli indici dei transformer block da cui prendere i token
        # poi dentro tokens_to_image fare il reshape corretto per ogni feature map come
        # richiesto dal decoder
        decoder_output = self.decoder([feature_map, skip_feature_map]) # [B, hidden_size, H, W]
        # Passaggio nella segmentation head
        logits = self.head(decoder_output) # [B, num_classes, H, W]
        # Upsampling finale
        upsampled_logits = nn.functional.interpolate(
            logits,
            size=x.shape[2:],  # dimensioni originali dell'input
            mode='bilinear',
            align_corners=False
        )
        
        return upsampled_logits # [B, num_classes, H, W]




    def adapt_patch_embed_to_channels(self, in_channels):
        """
        Adatta il patch embedding layer di DINOv3 per ricevere in input in_channels canali.
        inizializza i pesi dei canali aggiuntivi come la media dei pesi esistenti.

        Args:
            in_channels (int): Numero di canali di input desiderati
        """
        if in_channels == 3:
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