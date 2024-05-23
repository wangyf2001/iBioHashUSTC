def freeze_backbone(model, model_name, freeze_layer):
    if model_name == 'resnet50':
        for block in list(model.children())[:freeze_layer]:
            for param in list(block.parameters()):
                param.requires_grad = False
        print("{} has freeze {} blocks".format(model_name,freeze_layer))

    elif model_name == 'swin_large_patch4_window12_384':
        for block in list(model.children())[:2]:
            for param in list(block.parameters()):
                param.requires_grad = False
        
        for block in list(model.children())[2][:freeze_layer]:  # 1-4 可选参数
            for param in list(block.parameters()):
                param.requires_grad = False
        print("{} has freeze {} blocks".format(model_name,freeze_layer))

    elif model_name == 'beit_large_patch16_512.in22k_ft_in22k_in1k':
        for block in list(model.children())[:2]:
            for param in list(block.parameters()):
                param.requires_grad = False
        
        for block in list(model.children())[2][:freeze_layer]:  # 1-24 可选参数
            for param in list(block.parameters()):  
                param.requires_grad = False
        print("{} has freeze {} blocks".format(model_name,freeze_layer))

    elif model_name == 'vit_large_patch14_clip_336.openai_ft_in12k_in1k':
        for block in list(model.children())[:3]:
            for param in list(block.parameters()):
                param.requires_grad = False
        
        for block in list(model.children())[3][:freeze_layer]:  # 1-24 可选参数
            for param in list(block.parameters()):  
                param.requires_grad = False
        print("{} has freeze {} blocks".format(model_name,freeze_layer))

    elif model_name == 'tf_efficientnet_l2.ns_jft_in1k_475':
        for block in list(model.children())[:2]:
            for param in list(block.parameters()):
                param.requires_grad = False
        
        for block in list(model.children())[2][:freeze_layer]:  # 1-7 可选参数 
            for param in list(block.parameters()):  
                param.requires_grad = False
        print("{} has freeze {} blocks".format(model_name,freeze_layer))
    
    else:
        error = 'model freeze is not set'
        assert error == 'model freeze is set'

    
