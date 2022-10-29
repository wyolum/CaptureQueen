include <magnet.scad>
segments = 64;

module knight(){
rotate_extrude(convexity = 10, $fn = segments) {
  import(file = "profiles/my_knight_profile_0.svg");
}

intersection(){
rotate_extrude(convexity = 10, $fn = segments) {
  import(file = "profiles/my_knight_profile_1.svg");
}


color("red")translate([-14, 0, 22])rotate([90, 0, 0])translate([0, 0, -20])linear_extrude(height=40)import(file="profiles/my_knight_profile_2.svg");
color("green")translate([0, -9.5, 22])rotate([90, 0, 90])translate([0, 0, -20])linear_extrude(height=40)
  import(file="profiles/my_knight_profile_3.svg");
translate([-9, 0, 0])scale([2, 1, 1])rotate([0, 0, 45])translate([-8, -8, 16])cube([16, 16, 48]);
translate([6, 0, 0])scale([2, 1, 1])rotate([0, 0, 45])translate([-8, -8, 16])cube([16, 16, 48]);
}

difference(){
translate([1.5, 0, 42.])scale([3, 1, 1])rotate([0,0,60])cylinder($fn=3, d1=0, d2=14, h=10);
translate([-20, -5, 45])cube(10);
}
tab_handle();
}

difference(){
   union(){
        translate([0,0,lift])rotate(ROTATE)scale(SCALE)knight();
        if (lift != 0)
            translate([0, 0, 0])linear_extrude(height=lift)projection()rotate([90, 0, 0])scale(SCALE)knight();
    }
    //chop off the bottom in case we are rotated
    rotate([180,0,0])cylinder(h=King_Height,d=300);
    //magnet();
    king_magnet(o=.4*King_Height);
}
