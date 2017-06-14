//  x + 2y =  3
// 2x - 3y = -8 

// (1  2)(x)   ( 3)
// (2 -3)(y) = (-8)

// (x)   (1  2)^-1( 3)   (-1)
// (y) = (2 -3)   (-8) = ( 2)

// (1  2)^-1   (3/7   2/7)
// (2 -3)    = (2/7  -1/7)

function qerr = Q(abcd)
   a = abcd(1);
   b = abcd(2);
   c = abcd(3);
   d = abcd(4);

   e_1 = a + 2*c - 1;
   e_2 = b + 2*d;
   e_3 = 2*a - 3*c;
   e_4 = 2*b - 3*d - 1;

   qerr = e_1*e_1 + e_2*e_2 + e_3*e_3 + e_4*e_4;
endfunction

function abcd = update(abcd,lambda)
   a = abcd(1);
   b = abcd(2);
   c = abcd(3);
   d = abcd(4);

   e_1 = a + 2*c - 1;
   e_2 = b + 2*d;
   e_3 = 2*a - 3*c;
   e_4 = 2*b - 3*d - 1;

   nabla_a_Q = 2*e_1 + 4*e_3;
   nabla_b_Q = 2*e_2 + 4*e_4;
   nabla_c_Q = 4*e_1 - 6*e_3;
   nabla_d_Q = 4*e_2 - 6*e_4;

   nabla_Q = [nabla_a_Q, nabla_b_Q, nabla_c_Q, nabla_d_Q];

   abcd = abcd - lambda*nabla_Q;
endfunction

function abcd = params()
   abcd = [rand(),rand(),rand(),rand()];
   epsilon = 10^-6;
   lambda = 0.01;
   while Q(abcd) > epsilon*epsilon
      abcd = update(abcd,lambda);
   end
endfunction

function M = inverse_matrix()
   abcd = params()
   M(1,1) = abcd(1); M(1,2) = abcd(2);
   M(2,1) = abcd(3); M(2,2) = abcd(4);
endfunction

// M  =
// 
//    0.4285715    0.2857147  
//    0.2857143  - 0.1428570  

// M*[3;-8] = 
//
//  - 1.0000034  
//    1.9999986 



