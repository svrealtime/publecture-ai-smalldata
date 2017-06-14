// given 4 input-outputs of the boolean AND function
// determine the parameters of the best fitting function
// i.e. f(x,y) = axy + bx + cy + d
// via gradient descent

//  x y | x AND y
// -----+------------
//  0 0 |    0
//  0 1 |    0
//  1 0 |    0
//  1 1 |    1

xx = [0,1,0,1];
yy = [0,0,1,1];
xxyy = xx.*yy;
zz = [0,0,0,1];

// z = x AND y
//   = axy + bx + cy + d
//   = f(x,y)

// residual
// r_n = z_n - f(x_n,y_n)
//     = z_n - (ax_ny_n + bx_n + cy_n +d)

// sum of squared residuals
// Q(a,b,c,d) = Sum_n (z_n - f(x_n,y_n))^2

// Gradient
// see text

function [a, b, c, d] = update_one_step(a, b, c, d, lambda)  	// lambda is learning rate
   rr = a*xxyy + b*xx + c*yy + d - zz;				// (minus) residuals
   nabla_a = rr*xxyy';						// gradient w.r.t. a
   nabla_b = rr*xx';						// gradient w.r.t. b
   nabla_c = rr*yy';						// gradient w.r.t. c
   nabla_d = rr*ones(xx)';					// gradient w.r.t. d
   // update
   a = a - lambda*nabla_a;					// gradient descent
   b = b - lambda*nabla_b;
   c = c - lambda*nabla_c;
   d = d - lambda*nabla_d;
endfunction

function pr_gr_ss = progress(aa,bb,cc,dd)
   delta_aa = max(aa) - min(aa);
   delta_bb = max(bb) - min(bb);
   delta_cc = max(cc) - min(cc);
   delta_dd = max(dd) - min(dd);
   pr_gr_ss = sqrt(delta_aa*delta_aa + delta_bb*delta_bb + ...
                   delta_cc*delta_cc + delta_dd*delta_dd);
endfunction

function [a,b,c,d] = gradient_descent()
   aa = 1; bb = 1; cc = 1; dd = 1;				// initiate parameter sequences
   lambda = .01;   						// learning rate
   epsilon = 10^-6; 						// progress tolerance
   // fill sequences with 4 extra values
   for n = 1:4
      [a,b,c,d] = update_one_step(aa($), bb($), cc($), dd($), lambda);
      aa = [aa, a]; bb = [bb, b]; cc = [cc, c]; dd = [dd, d];
   end
   // continu if progress in not small enough
   while progress(aa,bb,cc,dd)>epsilon
      [a,b,c,d] = update_one_step(aa($), bb($), cc($), dd($), lambda);
      aa = [aa(2:$),a]; bb = [bb(2:$), b]; 
      cc = [cc(2:$),c]; dd = [dd(2:$), d];
   end
   // return tail values
   a = aa($);
   b = bb($);
   c = cc($);
   d = dd($);
endfunction

// calculate and report
[a,b,c,d] = gradient_descent();
disp('a = '+string(a));						// a = 0.9987606
disp('b = '+string(b));						// b = 0.0007660
disp('c = '+string(c));						// c = 0.0007660
disp('d = '+string(d));						// d = -0.0004734

// created function
function z = f(x,y)
   z = a*x*y + b*x + c*y + d;
endfunction

for y = 0:1
   for x = 0:1
      disp(string(x)+' AND '+string(y)+' = '+string(f(x,y)));
   end
end

// 0 AND 0 = -0.0004734   
// 1 AND 0 = 0.0002926   
// 0 AND 1 = 0.0002926   
// 1 AND 1 = 0.9998192 


