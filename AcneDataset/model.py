import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image

# Class definitions
CLASSES = ['Blackheads', 'Cyst', 'Papules', 'Pustules', 'Whiteheads']

# Clinical and Dermatological Knowledge Base
ACNE_KNOWLEDGE_BASE = {
    'Blackheads': {
        'medical_name': 'Open Comedones',
        'severity': 'Mild (Grade 1)',
        'severity_color': '#3b82f6', # Blue
        'severity_level': 1,
        'summary': 'Non-inflammatory acne formed when sebum (oil) and dead skin cells clog pores that remain open to the surface, causing oxidation and a darkened appearance.',
        'pathophysiology': 'Excess sebum production combined with abnormal follicular hyperkeratinization blocks the pilosebaceous canal. Air exposure oxidizes melanin in the plug, turning it dark.',
        'triggers': ['Excessive sebum/oil production', 'Hormonal fluctuations (androgens)', 'Comedogenic skincare products', 'High humidity & sweating', 'Improper facial cleansing'],
        'recommended_actives': [
            {'name': 'Salicylic Acid (BHA 1-2%)', 'purpose': 'Lipid-soluble acid that penetrates deep into pores to dissolve trapped sebum and cellular debris.'},
            {'name': 'Retinoids / Adapalene (0.1%)', 'purpose': 'Normalizes skin cell turnover and prevents microcomedone formation.'},
            {'name': 'Niacinamide (2-5%)', 'purpose': 'Regulates sebum synthesis, reduces pore visibility, and calms redness.'},
            {'name': 'Glycolic Acid (AHA 5-7%)', 'purpose': 'Exfoliates surface stratum corneum to prevent pore capping.'}
        ],
        'am_routine': [
            'Gentle foaming cleanser with 0.5-2% Salicylic Acid',
            'Lightweight Niacinamide serum (3-5%)',
            'Oil-free, non-comedogenic gel moisturizer (Hyaluronic Acid based)',
            'Broad-spectrum SPF 50+ mineral or matte chemical sunscreen'
        ],
        'pm_routine': [
            'Double cleanse: Gentle micellar water / cleansing oil followed by foaming cleanser',
            'Topical Adapalene 0.1% or Retinol (pea-sized amount on dry skin)',
            'Ceramide-rich barrier repair light moisturizer'
        ],
        'dos': ['Use non-comedogenic labeled makeup & moisturizers', 'Exfoliate with BHA 2-3 times weekly', 'Double cleanse in the evening if wearing sunscreen or makeup'],
        'donts': ['Do NOT use harsh physical pore strips frequently (can cause broken capillaries)', 'Do NOT squeeze or pick at pores (leads to enlarged pores and scarring)', 'Avoid heavy oil-based facial balms or coconut oil'],
        'when_to_see_doctor': 'If blackheads are persistent, widespread, or do not improve after 8-12 weeks of consistent OTC topical retinoid and salicylic acid use.'
    },
    'Whiteheads': {
        'medical_name': 'Closed Comedones',
        'severity': 'Mild to Moderate (Grade 1-2)',
        'severity_color': '#06b6d4', # Cyan
        'severity_level': 2,
        'summary': 'Non-inflammatory acne lesions where hair follicles are completely blocked by sebum and dead keratinocytes beneath a thin layer of skin.',
        'pathophysiology': 'Trapped sebum and cellular debris beneath the skin surface with no open orifice, appearing as small, flesh-colored to whitish bumps without active infection.',
        'triggers': ['Follicular hyperkeratosis', 'Heavy, occlusive skin creams or cosmetic foundations', 'Hormonal surges', 'Friction from phones, masks, or hats', 'Infrequent exfoliation'],
        'recommended_actives': [
            {'name': 'Adapalene / Tretinoin (Retinoids)', 'purpose': 'The gold standard for closed comedones; accelerates cell renewal and unclogs trapped microcomedones.'},
            {'name': 'Salicylic Acid (BHA 2%)', 'purpose': 'Penetrates lipid-rich follicle lining to dissolve keratinaceous plugs.'},
            {'name': 'Azelaic Acid (10-15%)', 'purpose': 'Keratolytic and anti-inflammatory properties with minimal irritation.'},
            {'name': 'Lactic Acid (5-10%)', 'purpose': 'Mild AHA to gently dissolve surface dead cells and hydrate skin.'}
        ],
        'am_routine': [
            'Mild hydrating cleanser (balanced pH 5.5)',
            'Azelaic Acid 10% suspension or Niacinamide serum',
            'Water-based lightweight oil-free lotion',
            'Broad-spectrum non-comedogenic SPF 50+'
        ],
        'pm_routine': [
            'Gentle foaming cleanser',
            'Adapalene 0.1% gel (apply evenly across affected areas)',
            'Soothe with centella asiatica / ceramide light barrier cream'
        ],
        'dos': ['Incorporate topical retinoids gradually (2-3 nights/week initially)', 'Keep pillowcases clean and washed weekly', 'Ensure all makeup is thoroughly removed at night'],
        'donts': ['Do NOT attempt to pop or needle whiteheads (causes infection, inflammation, and scarring)', 'Do NOT scrub skin aggressively', 'Avoid thick occlusive petroleum-based ointments on acne-prone zones'],
        'when_to_see_doctor': 'If closed comedones rapidly transition to inflamed red papules/pustules or remain resistant after 3 months of topical therapy.'
    },
    'Papules': {
        'medical_name': 'Inflammatory Papular Acne',
        'severity': 'Moderate (Grade 2-3)',
        'severity_color': '#f59e0b', # Amber
        'severity_level': 3,
        'summary': 'Inflamed, tender, raised reddish bumps without visible pus, caused by micro-ruptures in clogged follicular walls and proliferation of Cutibacterium acnes.',
        'pathophysiology': 'The blocked follicle ruptures intradermally, allowing sebum, bacteria (C. acnes), and keratin to spill into surrounding dermal tissue, triggering an active cellular immune response and erythema.',
        'triggers': ['C. acnes bacterial overgrowth', 'Picking or manipulating mild comedones', 'High glycemic diet / dairy intake', 'Stress and elevated cortisol', 'Hormonal surges (luteal phase)'],
        'recommended_actives': [
            {'name': 'Benzoyl Peroxide (2.5 - 5%)', 'purpose': 'Releases reactive oxygen species that eliminate C. acnes bacteria without causing bacterial resistance.'},
            {'name': 'Azelaic Acid (15-20%)', 'purpose': 'Potent anti-inflammatory agent that inhibits bacterial growth and prevents post-inflammatory erythema (PIE).'},
            {'name': 'Centella Asiatica / Madecassoside', 'purpose': 'Soothes intense dermal inflammation and supports barrier recovery.'},
            {'name': 'Topical Clindamycin (with BPO)', 'purpose': 'Targeted prescription antimicrobial therapy for inflammatory lesions.'}
        ],
        'am_routine': [
            'Gentle soothing cleanser',
            'Benzoyl Peroxide 2.5% gel (or Azelaic Acid 10-15%) applied to affected zones',
            'Barrier repair moisturizer with Panthenol / Centella',
            'Non-nano Mineral or chemical sunscreen SPF 50+'
        ],
        'pm_routine': [
            'Gentle pH-balanced gel cleanser',
            'Adapalene 0.1% or Retinoid (apply after skin is completely dry)',
            'Calming ceramide soothing cream'
        ],
        'dos': ['Use low-concentration Benzoyl Peroxide (2.5% is as effective as 10% with far less irritation)', 'Apply cold compresses if lesions are particularly tender', 'Use hydrocolloid soothing patches on sensitive spots'],
        'donts': ['Do NOT squeeze papules; there is no pus head to extract, and pressure forces inflammation deeper into dermis', 'Avoid combining multiple harsh acids simultaneously', 'Do NOT touch or rest hands on the face'],
        'when_to_see_doctor': 'If papules are widespread, painful, or leaving dark red/brown marks (PIE/PIH) after resolving.'
    },
    'Pustules': {
        'medical_name': 'Inflammatory Pustular Acne',
        'severity': 'Moderate to Severe (Grade 3)',
        'severity_color': '#f97316', # Orange
        'severity_level': 4,
        'summary': 'Inflamed red lesions with a distinct white or yellowish center filled with purulent fluid (pus), indicating significant immune cell recruitment to combat bacteria.',
        'pathophysiology': 'Neutrophils and inflammatory leukocytes migrate into the ruptured follicle, forming a localized purulent collection near the skin surface enclosed by an erythematous ring.',
        'triggers': ['Cutibacterium acnes colonization', 'Follicular rupture caused by touching/squeezing', 'Hormonal surges', 'Humid microenvironments under face coverings', 'Compromised skin lipid barrier'],
        'recommended_actives': [
            {'name': 'Benzoyl Peroxide (2.5 - 5%)', 'purpose': 'Rapid bactericidal action against C. acnes and reduces inflammatory exudate.'},
            {'name': 'Hydrocolloid Acne Patches (with Tea Tree/Salicylic)', 'purpose': 'Absorbs inflammatory fluid safely and prevents secondary bacterial contamination or picking.'},
            {'name': 'Azelaic Acid (15-20%)', 'purpose': 'Suppresses inflammatory cytokine cascades and prevents post-acne dark marks.'},
            {'name': 'Sulfur (3-5% spot wash)', 'purpose': 'Antiseptic and keratolytic agent that draws out impurities and dries surface pustules.'}
        ],
        'am_routine': [
            'Benzoyl Peroxide 2.5-4% face wash (leave on skin for 1-2 minutes before rinsing)',
            'Hydrating soothing serum with Hyaluronic acid & Niacinamide',
            'Lightweight non-comedogenic oil-free moisturizer',
            'Matte broad spectrum SPF 50+'
        ],
        'pm_routine': [
            'Gentle foaming cleanser',
            'Apply hydrocolloid pimple patches over active pustules',
            'Spot treatment with Azelaic Acid or Adapalene on surrounding acne zones',
            'Soothing barrier cream'
        ],
        'dos': ['Apply hydrocolloid pimple patches to protect and drain pustules hygienically', 'Wash face immediately after intense workouts or heavy sweating', 'Disinfect smartphone screens daily'],
        'donts': ['NEVER pop or burst pustules with unsterilized fingers or nails (leads to deep scarring and spreading)', 'Do NOT use abrasive facial scrubs over active pustules', 'Avoid drying out skin completely with alcohol toners'],
        'when_to_see_doctor': 'If pustules cover large areas of the face, neck, or back, or fail to respond to OTC antimicrobials within 6-8 weeks.'
    },
    'Cyst': {
        'medical_name': 'Nodulocystic Acne / Severe Cystic Acne',
        'severity': 'Severe (Grade 4 - Medical Attention Advised)',
        'severity_color': '#ef4444', # Red
        'severity_level': 5,
        'summary': 'Large, deep, painful, fluid-filled subcutaneous lesions formed deep within the dermis. High risk of permanent atrophic or hypertrophic scarring without medical intervention.',
        'pathophysiology': 'Severe follicular wall rupture deep in the dermis creates widespread perivascular inflammation, leading to large inflammatory cysts or interconnected nodules with extensive tissue destruction.',
        'triggers': ['Strong genetic predisposition', 'Excessive androgenic hormonal activity', 'Polycystic Ovary Syndrome (PCOS) / endocrine factors', 'Immune hyper-reactivity to C. acnes antigens', 'Severe psychological stress'],
        'recommended_actives': [
            {'name': 'Prescription Oral Isotretinoin', 'purpose': 'The most definitive systemic treatment; permanently reduces sebaceous gland size and sebum secretion.'},
            {'name': 'Prescription Oral Antibiotics (Doxycycline/Minocycline)', 'purpose': 'Reduces systemic inflammation and bacterial load in acute flareups.'},
            {'name': 'Hormonal Therapy (Spironolactone / Combined Oral Contraceptives)', 'purpose': 'Blocks androgen receptors in sebaceous glands for hormonal breakouts in females.'},
            {'name': 'Intralesional Corticosteroid Injections', 'purpose': 'Administered in-clinic by a dermatologist to rapidly flatten painful cysts within 24-48 hours.'},
            {'name': 'Topical Retinoids + Benzoyl Peroxide (Epiduo/Adapalene-BPO)', 'purpose': 'Prescription synergistic topical combo to maintain skin renewal and manage bacteria.'}
        ],
        'am_routine': [
            'Ultra-gentle hydrating foaming cleanser (no harsh sulfates)',
            'Centella Asiatica / Panthenol soothing serum',
            'Barrier recovery cream with Ceramides and Fatty Acids',
            'Non-irritating broad-spectrum SPF 50+ mineral sunscreen'
        ],
        'pm_routine': [
            'Gentle lipid-preserving facial cleanser',
            'Prescription topical medication as directed by Dermatologist',
            'Nourishing, non-comedogenic ceramide barrier balm / cream'
        ],
        'dos': ['Consult a board-certified dermatologist promptly to prevent permanent pitting scars', 'Apply ice wrapped in clean cloth for 5-10 minutes to reduce throbbing pain', 'Support skin barrier with gentle, non-irritating hydration'],
        'donts': ['ABSOLUTELY NEVER attempt to squeeze, lance, or press cysts (causes profound dermal scarring, tissue necrosis, and spreading)', 'Do NOT over-apply multiple harsh acid peeling solutions', 'Avoid picking or scrubbing'],
        'when_to_see_doctor': 'IMMEDIATELY. Cystic acne involves deep dermal layers and typically requires prescription oral medications (oral isotretinoin, spironolactone, or antibiotics) to resolve and prevent permanent scarring.'
    }
}

class AcneClassifier(nn.Module):
    """
    State-of-the-Art MobileNetV2 Transfer Learning Architecture with 
    customized head for Acne 5-class classification.
    """
    def __init__(self, num_classes=5, pretrained=True):
        super(AcneClassifier, self).__init__()
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        self.backbone = models.mobilenet_v2(weights=weights)
        
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes)
        )
        
        # Hook holders for Grad-CAM
        self.gradient = None
        self.activations = None
        
        # Register hooks on the last convolutional layer of backbone
        last_conv = self.backbone.features[-1]
        last_conv.register_forward_hook(self._forward_hook)
        last_conv.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradient = grad_output[0]

    def forward(self, x):
        return self.backbone(x)


def get_inference_transforms():
    """Standard ImageNet normalization and resize pipeline"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_train_transforms():
    """Data augmentation pipeline for robust acne classification"""
    return transforms.Compose([
        transforms.Resize((240, 240)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def compute_gradcam(model, image_tensor, target_class=None):
    """
    Computes Grad-CAM attention heatmap for explainable AI.
    Highlights regions of interest that influenced the model prediction.
    """
    model.eval()
    image_tensor = image_tensor.clone().detach().requires_grad_(True)
    output = model(image_tensor)
    
    if target_class is None:
        target_class = torch.argmax(output, dim=1).item()
        
    model.zero_grad()
    loss = output[0, target_class]
    loss.backward()
    
    # Pool gradients across channels
    gradients = model.gradient.detach().cpu().numpy()[0]
    activations = model.activations.detach().cpu().numpy()[0]
    
    weights = np.mean(gradients, axis=(1, 2))
    cam = np.zeros(activations.shape[1:], dtype=np.float32)
    
    for i, w in enumerate(weights):
        cam += w * activations[i]
        
    cam = np.maximum(cam, 0)
    if np.max(cam) > 0:
        cam = cam / np.max(cam)
    else:
        cam = np.zeros_like(cam)
        
    cam = cv2.resize(cam, (224, 224))
    return cam, target_class


def overlay_gradcam(pil_img, cam_map, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """
    Overlays Grad-CAM heatmap onto the original PIL image.
    """
    orig_np = np.array(pil_img.resize((224, 224)))
    if len(orig_np.shape) == 2:
        orig_np = cv2.cvtColor(orig_np, cv2.COLOR_GRAY2RGB)
    elif orig_np.shape[2] == 4:
        orig_np = cv2.cvtColor(orig_np, cv2.COLOR_RGBA2RGB)
        
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_map), colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    overlay = np.uint8(alpha * heatmap + (1 - alpha) * orig_np)
    return Image.fromarray(overlay), Image.fromarray(heatmap)
