include <lib.scad>

od = 19.7;
difference(){
  cylinder(d=24.25, h=15, $fn=50);
  translate([0,0, -1])cylinder(d=od, h=17);
}

/*
difference(){
  cylinder(h=3, d = 4.5, $fn=6);
  translate([0, 0, -.001])cylinder(h=.5, d1=1.5, d2=1, $fn=20);
  translate([0, 0, -1])cylinder(h=5, d=1, $fn=20);
}
*/
