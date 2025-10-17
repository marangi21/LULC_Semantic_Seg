import torch
import torch.nn as nn
from transformers import AutoModel
from segmentation_models_pytorch import DeepLabV3Plus
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

        deeplab_model = DeepLabV3Plus(
            encoder_name="resnet34",        # lo scarterò perché userò DINOv3 come backbone
            encoder_weights=None,           # non mi servono
            in_channels=self.hidden_size,   # dimensionalità dell'input del decoder
            classes=num_classes,         
        )

        self.decoder = deeplab_model.decoder

        self.head = deeplab_model.segmentation_head

    def forward(self, x):
        # x shape: [B, C, H, W]

        # Passaggio nel backbone, ottengo le hidden states (una per ognuno dei 40 transformer block di DINOv3)
        # questi token saranno trasformati dal neck in feature map (eventualmente piramidali) per il decoder
        backbone_outputs = self.backbone(x, output_hidden_states=True)
        hidden_states = backbone_outputs.hidden_states      # tuple di tensori, ognuno ha shape [B, num_tokens, hidden_size]
        last_hidden_state = hidden_states[-1]               # output finale del backbone, input per il modulo ASPP di DeeplabV3+

        # ToDo: implementare 4 feature map per comparabilità con gli esperimenti eseguiti con prithvi e terratorch
        # ToDo: serve un modo per far comunicare backbone e decoder. queste diventeranno le feature map per le skip connections             
        def tokens_to_image(hidden_states):
            skip_state_1 = hidden_states[5]             # uso l'output del 5°, 15°, 25° e 35° transformer block come skip connection per il decoder      
            skip_state_2 = hidden_states[15]            # servirebbe un modo per scegliere in maniera efficiente quali hidden states 
            skip_state_3 = hidden_states[25]            # usare come skip connections forse potrei utilizzare 10,20,30 e 40 per 
            skip_state_4 = hidden_states[35]            # matchare le 4 scale come negli esperimenti precedenti con terratorch
            B, N, C = skip_state_1.shape # batch size, num tokens, hidden size
            grid_size = int(math.sqrt(N)) # per creare l'immagine [sqrt(N), sqrt(N)]
            return hidden_state.permute(0, 2, 1).reshape(B, C, grid_size, grid_size)
        
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