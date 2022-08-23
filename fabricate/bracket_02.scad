module tube(od, id, h){
  difference(){
    color([1, .3, .6])cylinder(d=od, h=h);
    translate([0, 0, -1])cylinder(d=id, h=h + 2);
  }
}

OD = 20;
ID = 16;
angle = 20;


module leg(){
  color([.5, .1, 1])translate([0, 0 ,-90]){
    difference(){
      rotate([angle, 0, 0])translate([0, OD, 0])tube(OD, ID, 300);
      cylinder(h=1000, d=OD + 10);
      translate([-50, 0, 0])cube(100);
      translate([-100, -100, 300 * cos(angle)])cube(200);
    }
  }
}

module tripod(){
  leg();
  rotate([0, 0, 90]) leg();
  rotate([0, 0, 180]) leg();
  translate([0, 0, -240])tube(OD, ID, 300);
  translate([-250, 0, -375])rotate([0, 60, 0])tube(OD-5, ID-5, 300);
  color([.5, .5, .5])translate([0, 0, -30])tube(OD+10, ID+4, 70);
}

rotate([180, 0, 0])translate([0, 0, -190])tripod();
module board(square){
  for(i=[0:7]){
    for(j=[0:7]){
      if((i + j) % 2 == 1){
	color("grey")translate([i * square, j * square, 0])cube([square, square, 2]);
      }
      else{
	color("lightgrey")translate([i * square, j * square, 0])cube([square, square, 2]);
      }
    }
  }
}

inch = 25.4;
square = 2.25 * inch;
translate([-9, -4, 0] * square)board(2.25 * inch);

translate([1., -2.5, 9] * inch)rotate([0, -10, 0])cube([2, 5, 3] * inch);
