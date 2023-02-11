module magnet_cone(){
  translate([0, 0, -0.0]) union(){
    cylinder(d=12.7, h=3.5, $fn=50);
    translate([0, 0, 3.5])cylinder(d1=12.7, h=3, d2=6, $fn=50);
  }
}
