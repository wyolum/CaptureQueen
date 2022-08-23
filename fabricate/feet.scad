include <lib.scad>

OD = 24.3;
ID = 15;
inch = 25.4;

thickness = 5;

module leg(){
  tube(OD, ID, 100);
}

module front_foot(){
  difference(){
    union(){
      sphere(d=30);
      cylinder(d=30, h=inch);
    }
    leg();
  }
}

angle = 20;
module back_foot(){
  translate([0, 0, 15])rotate([angle, 0, 0])front_foot();
  translate([-40, -10, 0])cube([80, 30, 10]);
}

back_foot();
