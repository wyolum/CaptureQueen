segments = 64;

difference(){
  rotate_extrude(convexity = 10, $fn = segments) {
    import(file = "profiles/my_queen_profile.svg");
  }
  for(i=[1:8]){
    rotate([0, 0, i * 360/8])color("black")translate([0, 52, 0])rotate([30, 0, 0])scale([1, .25, 1])cylinder(d=12, h=200, $fn=30);
  }
}
