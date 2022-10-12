//designed to subtract from the base at origin

module magnet(bottom_offset = .2, magnet_dia=10.2, magnet_h=2, $fn=64){
    translate([0,0,bottom_offset])cylinder(h=magnet_h,d=magnet_dia);
}