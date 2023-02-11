$fn=64;
scale_factor = 50/88.;

module pawn(){
  scale(scale_factor)
  rotate_extrude(convexity = 10) {
    import(file = "profiles/pawn.svg");
  }
}

difference(){
  translate([0, 0, 1])rotate([90, 0, 0])pawn();
  translate([-50, -50, -100])cube(100);
}

