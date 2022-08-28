include <lib.scad>

$fn=50;

ribbon_width = 16.5 + 2;
H = 27;
D = 3 * inch;
thickness = 2;
d = D - 2 * thickness;

floor_h = 1;

module bottom(){
  //color("red")translate([0, 0, -floor_h])
  translate([D, 0, 0])
    difference(){
    cylinder(d=D, h=H);
    translate([0, 0, floor_h])cylinder(d=d, h=H);
    translate([-1, -100, 2 * floor_h])cube([2, 200, ribbon_width + floor_h]);
  }


}

module quartercircle(){
  color("green")
  translate([0, 0, 0])
    difference(){
    cylinder(d=d-2-5, h=H - floor_h);
    cylinder(d=d-2-5 - 2 * thickness, h=H - floor_h + 1);
    translate([-100, -198, -1])cube(200);
    translate([-225, 2, -1])cube(200);
  }
}

module top(){
  module semicircle(){
    color("purple")
      translate([d/4, 0, 0])
      difference(){
      cylinder(d=d/2-2-5, h=H - floor_h);
      translate([0, 0, -1])cylinder(d=d/2-2 * thickness-2-5, h=inch+2);
      translate([-100, -1, -100])cube(200);
    }
  }
  
  cylinder(d=D - 2 * thickness - 2, h=floor_h);
  semicircle();
  rotate([0, 0, 180])semicircle();
  
  quartercircle();
  rotate([0, 0, 180])quartercircle();

}
top();
