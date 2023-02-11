$fn=64;
scale_factor = 50/88.;

module knight(){
  scale(scale_factor)
    union(){
    difference(){
      rotate_extrude(convexity=10) {
	import(file = "profiles/my_knight_profile_0.svg");
      }
    }

    intersection(){
      rotate_extrude(convexity = 10) {
	import(file = "profiles/my_knight_profile_1.svg");
      }


      color("red")translate([-14, 0, 22])rotate([90, 0, 0])translate([0, 0, -20])
	linear_extrude(height=40)import(file="profiles/my_knight_profile_2.svg");
      color("green")translate([0, -9.5, 22])rotate([90, 0, 90])
	translate([0, 0, -20])linear_extrude(height=40)
	import(file="profiles/my_knight_profile_3.svg");
      translate([-9, 0, 0])scale([2, 1, 1])rotate([0, 0, 45])translate([-8, -8, 16])cube([16, 16, 48]);
      translate([6, 0, 0])scale([2, 1, 1])rotate([0, 0, 45])translate([-8, -8, 16])cube([16, 16, 48]);
    }

    difference(){
      translate([1.5, 0, 42.])scale([3, 1, 1])rotate([0,0,60])cylinder($fn=3, d1=0, d2=14, h=10);
      translate([-20, -5, 45])cube(10);
    }
  }
}

module half_knight(){
  difference(){
    translate([0, 0, 1])rotate([90, 0, 0])knight();
    translate([-50, -50, -100])cube(100);
  }
}

half_knight();
