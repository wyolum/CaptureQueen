OD = 24.5;

module tube(od, id, h){
  difference(){
    color([1, .3, .6])cylinder(d=od, h=h);
    translate([0, 0, -1])cylinder(d=id, h=h + 2);
  }
}

