include <lib.scad>

//OD = 24.3;
OD = 19.2;
thickness = 5;
cam_w = 25;
cam_h = 24;

L = 50;

module unit(){
  //translate([-6, -21, 0])rotate([0, 90, 0])import("camera_mount.stl");
  translate([0, 0, -(cam_w-10)/2])color("red")cylinder(d=10, h=cam_w - 10);

  translate([-5, 0, -(cam_w-10)/2])cube([10, L, cam_w -10]);
  translate([-7, 30, 0])rotate([-90, 0, 0])
    difference(){
    cylinder(d=OD + thickness, h=25);
    translate([0, 0, -1])cylinder(d=OD-5, h=102);
    translate([0, 0, 3])cylinder(d=OD, h=102);
  }
}

difference(){
  unit();
  translate([-7, 33, 0])rotate([-90, 0, 0])cylinder(d=OD, h=100);
  translate([0, 0, -50])color("red")cylinder(d=3.5, h=100, $fn=30);
}
