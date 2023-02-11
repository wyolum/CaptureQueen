inch = 25.4;

module screw(d){
  translate([0, 0, -15])color("blue")cylinder(h=30, d=d, $fn=20);
}

module m3_screws(){
    translate([-.9, -.9, 0] * inch)screw(3);
    translate([-.9,  .9, 0] * inch)screw(3);
    translate([.9,  -.9, 0] * inch)screw(3);
    translate([.9,  .9, 0] * inch)screw(3);

    translate([-.3, -.3, 0] * inch)screw(3);
    translate([-.3,  .3, 0] * inch)screw(3);
    translate([.3,  -.3, 0] * inch)screw(3);
    translate([.3,  .3, 0] * inch)screw(3);
}

module led(){
  translate([.15 * inch, .15 * inch, 0])rotate([0, 0, 45])translate([-2.5, -2.5, 0])cube([5, 5, 1.7]);
}

module ultim8x8(){
  difference(){
    union(){
      color("green")translate([-1.2 * inch, -1.2 * inch, 0])cube([2.4 * inch, 2.4 * inch, 1.6]);
      translate([-1.2 * inch, -1.2 * inch, 0])
      for(i=[0:7]){
	for(j=[0:7]){
	  translate([.3 * i * inch, .3 * j * inch, 1.6])led();
	}
      }
    }
    m3_screws();
  }
}

//ultim8x8();
translate([0, 0, -5])
rotate([0, 180, 0])
difference(){
  translate([0, (1.2 - .25) * inch, -10])difference(){
    cube([2.4 * inch, .5 * inch, 20], center=true);
    translate([0, 0, 5])cube([1.5 * inch, .5 * inch + 2, 15], center=true);
  }
  m3_screws();
}

cam_w = 25;
cam_h = 24;
difference(){
  translate([0, -.9 * inch, -2.5])cube([2.4 * inch, .5 * inch, 5], center=true);
  m3_screws();
}

xx = 33;
translate([0, -60, 0])
  difference(){
  union(){
    translate([-cam_w/2, 0, 0])rotate([0, 90, 0])cylinder(d=10, h=5, $fn=30);
    translate([+cam_w/2-5, 0, 0])rotate([0, 90, 0])cylinder(d=10, h=5, $fn=30);
    translate([0, xx/2, -2.5])cube([cam_w, xx, 5],center=true);
  }
  yy = 15;
  color("red")translate([-100, 0, 0])rotate([0, 90, 0])cylinder(d=3.5, h=200, $fn=30);
  color("red")translate([-yy/2, 0, 0])rotate([0, 90, 0])cylinder(d=10, h=yy, $fn=30);
  color("red")translate([-10-cam_w/2+4, 0, 0])rotate([0, 90, 0])cylinder(d=6.25, h=10, $fn=6);
  d1 = 2.4 * inch * sqrt(2) - 2;
  d2 = 2.8 * inch * sqrt(2);
  h = 20;
}
