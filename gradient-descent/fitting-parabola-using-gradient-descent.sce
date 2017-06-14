// given 6 samples (c.q. approximations) of a parabola
// determine the parameters of the best fitting parabola
// i.e. a, b, and c in y = ax^2 + bx + c
// using gradient descent
// e.g. y=x^2-4x+3

xx = [-1.0,  0.1,  1.1,  1.9,  2.9, 4.1];
yy = [ 8.5,  2.5, -0.2, -0.7, -0.1, 3.2];

function [a, b, c] = update_one_step(a, b, c, xx, yy, lambda)  	// lambda is learning rate
   rr = a*(xx.*xx) + b*xx + c - yy;				// (minus) residuals
   nabla_a = rr*(xx.*xx)';					// gradient w.r.t. a
   nabla_b = rr*xx';						// gradient w.r.t. b
   nabla_c = rr*ones(xx)';					// gradient w.r.t. c
   // update
   a = a - lambda*nabla_a;					// gradient descent
   b = b - lambda*nabla_b;
   c = c - lambda*nabla_c;
endfunction

function pr_gr_ss = progress(aa,bb,cc)
   delta_aa = max(aa) - min(aa);
   delta_bb = max(bb) - min(bb);
   delta_cc = max(cc) - min(cc);
   pr_gr_ss = sqrt(delta_aa*delta_aa + delta_bb*delta_bb + delta_cc*delta_cc);
endfunction

function [a,b,c] = gradient_descent(xx,yy)
   aa = 1;							// initiate sequence of a's
   bb = 0;							// initiate sequence of b's
   cc = 0;							// initiate sequence of c's
   lambda = .001;   						// learning rate
   epsilon = 10^-6; 						// progress tolerance
   // fill sequences aa and bb with 4 extra values
   for n = 1:4
      [a,b,c] = update_one_step(aa($), bb($), cc($), xx, yy, lambda);
      aa = [aa, a]; bb = [bb, b]; cc = [cc, c];
   end
   // continu if progress in not small enough
   while progress(aa,bb,cc)>epsilon
      [a,b,c] = update_one_step(aa($), bb($), cc($), xx, yy, lambda);
      aa = [aa(2:$),a]; bb = [bb(2:$), b]; cc = [cc(2:$), c];
   end
   // return tail values
   a = aa($);
   b = bb($);
   c = cc($);
endfunction

// calculate and report
[a,b,c] = gradient_descent(xx,yy);
disp('a = '+string(a));						// a = 1.0061957
disp('b = '+string(b));						// b = -4.1006259 
disp('c = '+string(c));						// c = 3.21219

// define function
function y = f(x)
   y = a.*x.*x + b.*x + c;
endfunction

// plot
clf()
CROSS = -2;
plot2d(xx,yy,CROSS);
x_min = min(xx); x_max = max(xx);
RED = 5;
x_range = x_min:.01:x_max;
plot2d(x_range,f(x_range),RED);
xtitle('fitting parabola, using Gradient Descent');
xlabel('x'); ylabel('y');
