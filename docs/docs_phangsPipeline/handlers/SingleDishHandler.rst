#################
SingleDishHandler
#################

The SingleDishHandler allows for imaging of singledish ALMA data.
This essentially replicates what the ALMA pipeline does, and then
produces ``.image`` and ``.weight`` cubes at the end. The weight
cube scales roughly as :math:`t_int \times \Delta v / T_sys^2`,
and so are squared afterwards, so they can be fed directly
into later linear mosaicking.

.. autofunction:: phangsPipeline.SingleDishHandler.loop_singledish
    :noindex:
