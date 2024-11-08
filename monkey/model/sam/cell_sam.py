from segment_anything import (
    sam_model_registry,
    SamAutomaticMaskGenerator,
    SamPredictor,
)
from pprint import pprint
from torchvision.transforms import v2
import torch

if __name__ == "__main__":
    checkpoint_path = "/home/u1910100/cloud_workspace/data/Monkey/SAM_weights/sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    model = sam_model_registry[model_type](checkpoint=checkpoint_path)
    model.to('cuda')
    pprint(model)

    image = torch.ones(size=(1,3,256,256),device='cuda', dtype=torch.float32)

    transforms = v2.Compose([
        v2.Resize((1024,1024))
    ])

    image = transforms(image)

    print(image.size())

    model.eval()

    image_embedding = model.image_encoder(image)
    sparse_embeddings, dense_embeddings = model.prompt_encoder(
        points=None,boxes = None,masks = None)
    
    gland_low_res_masks, gland_iou_predictions = model.mask_decoder(
        image_embeddings=image_embedding,
        image_pe=model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )

    gland_masks = model.postprocess_masks(
        gland_low_res_masks,
        input_size=((1024,1024)),
        original_size=((256,256)),
    )

    gland_masks = torch.sigmoid(gland_masks)

    print(gland_masks.size())