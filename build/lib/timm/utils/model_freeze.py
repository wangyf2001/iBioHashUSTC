def freeze_backbone(model, model_name, freeze_layer):
    if model_name = 'resnet50':
        for block in list(model.children())[:freeze_layer]:
            for param in list(block.parameters()):
                param.requires_grad = False
    
    if model_name = 'swin_large_patch4_window12_384':
        for block in list(model.children())[:2]:
            for param in list(block.parameters()):
                param.requires_grad = False
        
        for block in list(model.children())[2][:freeze_layer]:
            for param in list(block.parameters()):
                param.requires_grad = False

    
