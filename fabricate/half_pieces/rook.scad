$fn=64;
scale_factor = 50/88.;
module rook(){
  scale(scale_factor);
  difference(){
    rotate_extrude(convexity = 10) {
      import(file = "profiles/my_rook_profile.svg");
    }
    color("red")translate([-50, -1.5, 46])cube([100, 3, 10]);
    rotate([0, 0, 90])color("red")translate([-50, -1.5, 46])cube([100, 3, 10]);
  }
}

module half_rook(){
  difference(){
    translate([0, 0, 1])rotate([90, 0, 0])rook();
    translate([-50, -50, -100])cube(100);
  }
}

half_rook();
