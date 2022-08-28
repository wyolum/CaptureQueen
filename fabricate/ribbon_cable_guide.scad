include <lib.scad>

H = 10;
thickness = 3;

difference(){
  union(){
    cylinder(d=snug_id_small + 2 * thickness, h=H);
    translate([-(snug_id_small + 2 * thickness) / 2, 0, 0])cube([snug_id_small + 2 * thickness, 25, H]);
  }
  translate([0, 0, -1])cylinder(d=snug_id_small - .5, h=H+2);
  translate([-snug_id_small / 2, 12, -1])cube([snug_id_small, 25, H + 2]);
#translate([0, -70, 0])rotate([0, 0, 45])translate([-50, -50, -1])cube([100, 100, H+2]);
}
