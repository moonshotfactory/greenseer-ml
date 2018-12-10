import gluoncv as gcv
from gluoncv.utils import download, viz

classes = ['bishop', 'rook']

net = gcv.model_zoo.get_model('ssd_512_mobilenet1.0_custom', classes=classes, pretrained_base=False)
net.load_parameters('ssd_512_mobilenet1.0_chess.params')
x, image = gcv.data.transforms.presets.ssd.load_test('real_test.jpg', 512)
cid, score, bbox = net(x)
ax = viz.plot_bbox(image, bbox[0], score[0], cid[0], class_names=classes)
print(type(ax))
ax.figure.savefig('result_real_test.pdf')