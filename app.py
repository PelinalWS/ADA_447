import PIL
from fastai.vision.all import load_learner, PILImage
import gradio as gr

learn = load_learner('./models/resnet34-learn4.pkl')

def predict(img):
    if not isinstance(img, PIL.Image.Image):
        img = PILImage.create(img)
        img = img.resize((224, 224))
    pred_class, pred_idx, probs = learn.predict(img)
    class_names = learn.dls.vocab
    return {
        "predicted_class": str(pred_class),
        "confidence": round(probs[pred_idx].item(), 4),
        "all_probabilities": {
            str(cls): round(probs[i].item(), 4)
            for i, cls in enumerate(class_names)
        }
    }

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=2),
    title="Traffic Sign and Light Recognition",
    description="."
)