\ LP format - for model browsing. Use MPS format to capture full model detail.
\ Signature: 0x3b9be1738b7ffedf
Maximize
  0.0434782608695652 d_xs[K8076324915] - p_f - Constant
Subject To
 cons_veh_cap: 260 d_xs[N9412683057] + 180 d_xs[G0621548397]
   + 270 d_xs[T3412985670] + 210 d_xs[V6314209875] + 240 d_xs[P3865270194]
   + 170 d_xs[K8076324915] <= 8000
 cons_veh_min_load: 260 d_xs[N9412683057] + 180 d_xs[G0621548397]
   + 270 d_xs[T3412985670] + 210 d_xs[V6314209875] + 240 d_xs[P3865270194]
   + 170 d_xs[K8076324915] + 99999 p_f >= 5800
 cons_xs_bounds[N9412683057]: d_xs[N9412683057] <= 18
 cons_xs_bounds[G0621548397]: d_xs[G0621548397] <= 18
 cons_xs_bounds[T3412985670]: d_xs[T3412985670] <= 19
 cons_xs_bounds[V6314209875]: d_xs[V6314209875] <= 16
 cons_xs_bounds[P3865270194]: d_xs[P3865270194] <= 6
 cons_xs_bounds[K8076324915]: d_xs[K8076324915] <= 23
Bounds
 Constant = 1
Binaries
 p_f
Generals
 d_xs[N9412683057] d_xs[G0621548397] d_xs[T3412985670] d_xs[V6314209875]
 d_xs[P3865270194] d_xs[K8076324915]
End
