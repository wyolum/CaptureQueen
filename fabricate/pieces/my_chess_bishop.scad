segments = 64;
include<lib.scad>

difference(){
  scale([.9, .9, 1])rotate_extrude(convexity = 10, $fn = segments) {
    import(file = "profiles/my_bishop_profile.svg");
  }
  color("red")translate([-50, -3, 48])rotate([25, 0, 0])cube([100, 2, 10]);
  translate([0, 0, -.01])magnet_cone();
}

