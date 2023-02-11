segments = 64;
include("lib.scad")

module pawn(){
  rotate_extrude(convexity = 10, $fn = segments) {
    import(file = "profiles/my_pawn_profile.svg");
  }
}

difference(){
  translate([0, 0, 0.1])pawn();
  magnet_cone();
}
