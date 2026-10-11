\ LP format - for model browsing. Use MPS format to capture full model detail.
\ Signature: 0x38ef9be1200c1d02
Minimize
  5000 d_alpha[Z4016328,D4370925168] + 5000 d_alpha[Z4016328,G0732841956]
   + 5000 d_alpha[Z4016328,W0257169348]
   + 5000 d_alpha[P6482315,D4370925168]
   + 5000 d_alpha[P1624079,C7421096835]
   + 5000 d_alpha[P1624079,D4370925168]
   + 5000 d_alpha[P1624079,Z7692481305]
   + 5000 d_alpha[P6482315,Z7692481305]
   + 5000 d_alpha[Q2896305,Y0318265794]
   + 5000 d_alpha[P1624079,G0732841956]
   + 5000 d_alpha[P1624079,W0257169348]
   + 5000 d_alpha[Q2896305,O9137586024]
   + 5000 d_alpha[P6482315,G0732841956]
   + 5000 d_alpha[P6482315,W0257169348]
   + 5000 d_alpha[B2183649,Y0318265794]
   + 5000 d_alpha[B2183649,O9137586024]
   + 5000 d_alpha[Q2896305,H0381726459]
   + 5000 d_alpha[B2183649,K2346097185]
   + 5000 d_alpha[Z4016328,Y0318265794]
   + 5000 d_alpha[Z4016328,O9137586024]
   + 5000 d_alpha[Z4016328,K2346097185]
   + 5000 d_alpha[P1624079,Y0318265794]
   + 5000 d_alpha[P6482315,Y0318265794]
   + 5000 d_alpha[P1624079,K2346097185]
   + 5000 d_alpha[P1624079,O9137586024]
   + 5000 d_alpha[P6482315,V1902634578]
   + 5000 d_alpha[P6482315,O9137586024]
   + 5000 d_alpha[P1624079,H0381726459]
   + 5000 d_alpha[Q2896305,D4370925168]
   + 5000 d_alpha[P6482315,H0381726459]
   + 5000 d_alpha[Q2896305,C7421096835]
   + 5000 d_alpha[Q2896305,Z7692481305]
   + 5000 d_alpha[B2183649,D4370925168]
   + 5000 d_alpha[B2183649,C7421096835]
   + 5000 d_alpha[B2183649,Z7692481305]
   + 5000 d_alpha[Q2896305,G0732841956]
   + 5000 d_alpha[Q2896305,W0257169348]
   + 5000 d_alpha[B2183649,G0732841956]
   + 5000 d_alpha[B2183649,W0257169348] + 5000 d_beta[W0257169348]
   + 5000 d_beta[G0732841956] + 5000 d_beta[H0381726459]
   + 5000 d_beta[Y0318265794] + 5000 d_beta[V1902634578]
   + 5000 d_beta[Z7692481305] + 5000 d_beta[C7421096835]
   + 5000 d_beta[K2346097185] + 5000 d_beta[D4370925168]
   + 5000 d_beta[O9137586024] + 5000 d_gamma[9521,V1902634578]
   + 5000 d_gamma[9521,C7421096835] + 5000 d_gamma[9521,O9137586024]
   + 5000 d_gamma[9521,Z7692481305] + 5000 d_gamma[4666,H0381726459]
   + 5000 d_gamma[9521,K2346097185] + 5000 d_gamma[9521,H0381726459]
   + 5000 d_gamma[4666,G0732841956] + 5000 d_gamma[4666,Z7692481305]
   + 5000 d_gamma[9521,G0732841956] + 5000 d_gamma[9521,W0257169348]
   + 5000 d_gamma[4666,Y0318265794] + 5000 d_gamma[4666,D4370925168]
   + 5000 d_gamma[9521,Y0318265794] + 5000 d_gamma[4666,K2346097185]
   + 5000 d_gamma[4666,O9137586024]
Subject To
 cons_dealer_demands[Z4016328,D4370925168]: d_alpha[Z4016328,D4370925168]
   >= 8
 cons_dealer_demands[Z4016328,G0732841956]: d_alpha[Z4016328,G0732841956]
   >= 21
 cons_dealer_demands[Z4016328,W0257169348]: d_alpha[Z4016328,W0257169348]
   >= 11
 cons_dealer_demands[P6482315,D4370925168]: d_alpha[P6482315,D4370925168]
   >= 11
 cons_dealer_demands[P1624079,C7421096835]: d_alpha[P1624079,C7421096835]
   >= 6
 cons_dealer_demands[P1624079,D4370925168]: d_alpha[P1624079,D4370925168]
   >= 12
 cons_dealer_demands[P1624079,Z7692481305]: d_alpha[P1624079,Z7692481305]
   >= 13
 cons_dealer_demands[P6482315,Z7692481305]: d_alpha[P6482315,Z7692481305]
   >= 17
 cons_dealer_demands[Q2896305,Y0318265794]: d_alpha[Q2896305,Y0318265794]
   >= 7
 cons_dealer_demands[P1624079,G0732841956]: d_alpha[P1624079,G0732841956]
   >= 21
 cons_dealer_demands[P1624079,W0257169348]: d_alpha[P1624079,W0257169348]
   >= 11
 cons_dealer_demands[Q2896305,O9137586024]: d_alpha[Q2896305,O9137586024]
   >= 17
 cons_dealer_demands[P6482315,G0732841956]: d_alpha[P6482315,G0732841956]
   >= 16
 cons_dealer_demands[P6482315,W0257169348]: d_alpha[P6482315,W0257169348]
   >= 16
 cons_dealer_demands[B2183649,Y0318265794]: d_alpha[B2183649,Y0318265794]
   >= 7
 cons_dealer_demands[B2183649,O9137586024]: d_alpha[B2183649,O9137586024]
   >= 19
 cons_dealer_demands[Q2896305,H0381726459]: d_alpha[Q2896305,H0381726459]
   >= 19
 cons_dealer_demands[B2183649,K2346097185]: d_alpha[B2183649,K2346097185]
   >= 3
 cons_dealer_demands[Z4016328,Y0318265794]: d_alpha[Z4016328,Y0318265794]
   >= 4
 cons_dealer_demands[Z4016328,O9137586024]: d_alpha[Z4016328,O9137586024]
   >= 13
 cons_dealer_demands[Z4016328,K2346097185]: d_alpha[Z4016328,K2346097185]
   >= 11
 cons_dealer_demands[P1624079,Y0318265794]: d_alpha[P1624079,Y0318265794]
   >= 12
 cons_dealer_demands[P6482315,Y0318265794]: d_alpha[P6482315,Y0318265794]
   >= 6
 cons_dealer_demands[P1624079,K2346097185]: d_alpha[P1624079,K2346097185]
   >= 16
 cons_dealer_demands[P1624079,O9137586024]: d_alpha[P1624079,O9137586024]
   >= 4
 cons_dealer_demands[P6482315,V1902634578]: d_alpha[P6482315,V1902634578]
   >= 8
 cons_dealer_demands[P6482315,O9137586024]: d_alpha[P6482315,O9137586024]
   >= 15
 cons_dealer_demands[P1624079,H0381726459]: d_alpha[P1624079,H0381726459]
   >= 14
 cons_dealer_demands[Q2896305,D4370925168]: d_alpha[Q2896305,D4370925168]
   >= 4
 cons_dealer_demands[P6482315,H0381726459]: d_alpha[P6482315,H0381726459]
   >= 10
 cons_dealer_demands[Q2896305,C7421096835]: d_alpha[Q2896305,C7421096835]
   >= 17
 cons_dealer_demands[Q2896305,Z7692481305]: d_alpha[Q2896305,Z7692481305]
   >= 17
 cons_dealer_demands[B2183649,D4370925168]: d_alpha[B2183649,D4370925168]
   >= 5
 cons_dealer_demands[B2183649,C7421096835]: d_alpha[B2183649,C7421096835]
   >= 3
 cons_dealer_demands[B2183649,Z7692481305]: d_alpha[B2183649,Z7692481305]
   >= 17
 cons_dealer_demands[Q2896305,G0732841956]: d_alpha[Q2896305,G0732841956]
   >= 29
 cons_dealer_demands[Q2896305,W0257169348]: d_alpha[Q2896305,W0257169348]
   >= 15
 cons_dealer_demands[B2183649,G0732841956]: d_alpha[B2183649,G0732841956]
   >= 12
 cons_dealer_demands[B2183649,W0257169348]: d_alpha[B2183649,W0257169348]
   >= 22
 cons_sku_supply[W0257169348]: d_beta[W0257169348] >= -123
 cons_sku_supply[G0732841956]: d_beta[G0732841956] >= -125
 cons_sku_supply[H0381726459]: d_beta[H0381726459] >= -89
 cons_sku_supply[Y0318265794]: d_beta[Y0318265794] >= -82
 cons_sku_supply[V1902634578]: d_beta[V1902634578] >= -42
 cons_sku_supply[Z7692481305]: d_beta[Z7692481305] >= -102
 cons_sku_supply[C7421096835]: d_beta[C7421096835] >= -59
 cons_sku_supply[K2346097185]: d_beta[K2346097185] >= -80
 cons_sku_supply[D4370925168]: d_beta[D4370925168] >= -52
 cons_sku_supply[O9137586024]: d_beta[O9137586024] >= -109
 cons_fact_sku_inv[9521,V1902634578]: d_gamma[9521,V1902634578] >= -42
 cons_fact_sku_inv[9521,C7421096835]: d_gamma[9521,C7421096835] >= -59
 cons_fact_sku_inv[9521,O9137586024]: d_gamma[9521,O9137586024] >= -49
 cons_fact_sku_inv[9521,Z7692481305]: d_gamma[9521,Z7692481305] >= -14
 cons_fact_sku_inv[4666,H0381726459]: d_gamma[4666,H0381726459] >= -65
 cons_fact_sku_inv[9521,K2346097185]: d_gamma[9521,K2346097185] >= -37
 cons_fact_sku_inv[9521,H0381726459]: d_gamma[9521,H0381726459] >= -24
 cons_fact_sku_inv[4666,G0732841956]: d_gamma[4666,G0732841956] >= -69
 cons_fact_sku_inv[4666,Z7692481305]: d_gamma[4666,Z7692481305] >= -88
 cons_fact_sku_inv[9521,G0732841956]: d_gamma[9521,G0732841956] >= -56
 cons_fact_sku_inv[9521,W0257169348]: d_gamma[9521,W0257169348] >= -123
 cons_fact_sku_inv[4666,Y0318265794]: d_gamma[4666,Y0318265794] >= -31
 cons_fact_sku_inv[4666,D4370925168]: d_gamma[4666,D4370925168] >= -52
 cons_fact_sku_inv[9521,Y0318265794]: d_gamma[9521,Y0318265794] >= -51
 cons_fact_sku_inv[4666,K2346097185]: d_gamma[4666,K2346097185] >= -43
 cons_fact_sku_inv[4666,O9137586024]: d_gamma[4666,O9137586024] >= -60
Bounds
End
