include <lib.scad>

ID = 20;
angle = 20;
thickness=6;

module leg(){
  rotate([-angle, 0, 0])translate([0, OD+1, -9])cylinder(d=OD, h = 100);
  rotate([-angle, 0, 0])translate([0, OD+1, -99])cylinder(d=15, h = 200);
}
module insert(){
  color("red")
  rotate([-angle, 0, 0])translate([0, OD+1, -70])cylinder(d=OD + thickness, h = 100);
}

module legs(){
  leg();
  rotate([0, 0, 180])leg();
  rotate([0, 0, 90])leg();
}
module inserts(){
  insert();
  rotate([0, 0, 180])insert();
  rotate([0, 0, 90])insert();
}

module manifold(){
  translate([0, 0, -90])
    difference(){
    cylinder(h=70, d=OD + thickness);
    translate([0, 0, -thickness])cylinder(h=70, d=OD);
  }  
  difference(){
    inserts();
    legs();
  }
}

difference(){
  manifold();
  //translate([0, 0, -120-thickness])cylinder(h=110, d=OD);
translate([0, 0, -130-thickness])cylinder(h=110, d=OD);
translate([0, 0, -70])cylinder(h=110, d=ID);
legs();
translate([-500, -500, 4])cube(1000);
translate([-500, -500, -1070])cube(1000);
}

