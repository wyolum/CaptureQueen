segments = 64;

module king(){
  rotate_extrude(convexity = 10, $fn = segments) {
    import(file = "profiles/my_king_profile.svg");
  }

  translate([-6.75, 0, 73.86])rotate([90, 0, 0])translate([0, 0, -2])linear_extrude(height=4)
    import(file="my_king_cross.svg");
}
king();
