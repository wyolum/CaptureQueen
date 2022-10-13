/* 
    Edit for default chess set parameters
*/
// Don't change these unneccesarily
inch = 25.4;
// measured height of the king as designed/no scaling
DEFAULT_KING_HEIGHT = 88;


//Parameters based on set scaling and magnet size
//DEFAULT_MAGNET_DIA=10.2;
//DEFAULT_MAGNET_H=2;
DEFAULT_MAGNET_DIA=6.1;
DEFAULT_MAGNET_H=2;

// this is how far into the piece the magnet is "lifted" zero for a hole in the bottom,
// .2 gives one layer of plastic for my default printing parameters
DEFAULT_BOTTOM_OFFSET =.2;
//DEFAULT_BOTTOM_OFFSET =0;

// if you want to scale the whole set do it relative to the king
// King_Height = DEFAULT_KING_HEIGHT for no scaeing
King_Height = 2*inch;

//whole pieces standing up or half pieces laying down.
Half_Pieces = true;
// one magnet or two for the lying down pieces?
Two_magnets = true;

//Don't change (calculations based on variables above)
Scale_factor = King_Height/DEFAULT_KING_HEIGHT;
SCALE=[Scale_factor,Scale_factor,Scale_factor];
ROTATE = (Half_Pieces) ? [90,0,0] : [0,0,0];

    