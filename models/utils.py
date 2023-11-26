import timm
import torch


def calculate_feature_filters(arch):
    feature_extractor = timm.create_model(arch, pretrained=False, features_only=True)
    dummy_input = torch.randn((1, 3, 224, 224))

    try:
        feat0, feat1, feat2, feat3, feat4 = feature_extractor(dummy_input)

        feat0_filters = feat0.size(1)
        feat1_filters = feat1.size(1)
        feat2_filters = feat2.size(1)
        feat3_filters = feat3.size(1)
        feat4_filters = feat4.size(1)

        feature_filters = [feat0_filters, feat1_filters, feat2_filters, feat3_filters, feat4_filters]
    except ValueError:  # Might have only 4 features
        feat0, feat1, feat2, feat3 = feature_extractor(dummy_input)

        feat0_filters = feat0.size(1)
        feat1_filters = feat1.size(1)
        feat2_filters = feat2.size(1)
        feat3_filters = feat3.size(1)

        feature_filters = [feat0_filters, feat1_filters, feat2_filters, feat3_filters]

    del feature_extractor

    return feature_filters
