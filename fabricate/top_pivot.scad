include <lib.scad>

inch = 25.4;
ID = 19.5;

thickness = 3;
OD = ID + 2 * thickness;

H = inch;

module half(){
  difference(){
    union(){
      tube(OD, ID, H);
      translate([0, -inch/2, 0])cube([.75 * inch, OD, H]);
      translate([inch/2, -ID/2 - thickness, 0])cube([inch, thickness, H]);
      translate([1.5 * inch, -ID/2, inch/2])rotate([90, 0, 0])
	cylinder(d=inch, h=thickness);
    }
    union(){
      translate([0, 0, -1])cylinder(h=H + 2, d=ID);
      //#translate([2 * inch, 0, -1])rotate([0, -40, 0])cylinder(h=100, d=ID);
      translate([1.5 * inch, 10,H/2])rotate([90, 0, 0])cylinder(d=3, h=100, $fn=30);
      translate([32, -ID/2, 0])rotate([0, -40, 0])cube([1.5 * ID, ID, 100]);
    }
  }
}

half();
mirror([0, 1, 0])half();

