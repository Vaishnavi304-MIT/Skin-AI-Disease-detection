import os
import timm

# Clean up invalid SSL_CERT_FILE variable if the path doesn't exist on disk
if "SSL_CERT_FILE" in os.environ and not os.path.exists(os.environ["SSL_CERT_FILE"]):
    del os.environ["SSL_CERT_FILE"]

try:
    import certifi
    os.environ["SSL_CERT_FILE"] = certifi.where()
except ImportError:
    pass

def build_mobilevit(num_classes: int, pretrained: bool = True):
    model = timm.create_model('mobilevit_s', pretrained=pretrained, num_classes=num_classes)
    return model