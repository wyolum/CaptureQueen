include <lib.scad>

ID = 15;
inch = 25.4;

thickness = 5;

module leg(){
  tube(OD, ID, 100);
}

module foot(){
  difference(){
    union(){
      sphere(d=30);
      cylinder(d=30, h=10);
    }
    leg();
translate([-100, -100, 10])cube(200);
  }
}

foot();
