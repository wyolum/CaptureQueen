hole_w = 21;
hole_h = 12.5;

cam_w = 25;
cam_h = 24;

square = 9;
h = 5;

module screw(){
  translate([0, 0, -3.5])color("red")cylinder(h=10, d=1.5, $fn=20);
}
module screws(){
  translate([hole_w/2,  hole_h/2, -1])screw();
  translate([hole_w/2, -hole_h/2, -1])screw();
  translate([-hole_w/2,  hole_h/2, -1])screw();
  translate([-hole_w/2, -hole_h/2, -1])screw();
}

//color("gray")translate([0, 3.75, 0])import("RPi_Camera_V2.1.stl");

up = 6;

difference(){
  union(){
    translate([0, 4, 1])color("purple")translate([-cam_w / 2, -cam_h / 2, 0])cube([cam_w, cam_h + 5, h]);
    translate([-cam_w/2, 21, up])rotate([0, 90, 0])cylinder(d=10, h=5, $fn=30);
    translate([+cam_w/2-5, 21, up])rotate([0, 90, 0])cylinder(d=10, h=5, $fn=30);
  }
  
  translate([0, 5.5, 3])color([.3, .5, .2])scale([cam_w, cam_h, 1])rotate([0, 0, 45])cylinder(h=4, d2=1.3, d1=.5, $fn=4);

  translate([-4.55, 1.8, 0])color("red")cube([square, square, 100]);
  translate([-4.55-2.5, -11.8+5, .9])color("green")cube([14, 16, 2.1]);
 
color("red")translate([-(cam_w - 10) / 2, 21, up])rotate([0, 90, 0])cylinder(d=10, h=cam_w-10, $fn=30);
color("red")translate([-100, 21, up])rotate([0, 90, 0])cylinder(d=3.5, h=200, $fn=30);
color("red")translate([-10-cam_w/2+4, 21, up])rotate([0, 90, 0])cylinder(d=6.25, h=10, $fn=6);

  screws();
}


