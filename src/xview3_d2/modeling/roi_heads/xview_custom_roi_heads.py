from detectron2.modeling.roi_heads.roi_heads import ROI_HEADS_REGISTRY, StandardROIHeads

from .xview_custom_fast_rcnn import xViewFastRCNNOutputLayers

__all__ = ["xViewStandardROIHeads"]
@ROI_HEADS_REGISTRY.register()
class xViewStandardROIHeads(StandardROIHeads):
    @classmethod
    def _init_box_head(self, cfg, input_shape):
        ret = super()._init_box_head(cfg, input_shape)
        del ret["box_predictor"]
        ret["box_predictor"] = xViewFastRCNNOutputLayers(
            cfg, ret["box_head"].output_shape
        )
        return ret
