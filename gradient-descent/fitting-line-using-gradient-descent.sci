// given 5 samples (c.q. approximated points) of a line
// determine the parameters of the best fitting line
// i.e. a and b in l:y = ax + b
// via gradient descent
// e.g. y=2x-3

xx = 1:5;
yy = [-1.1, 0.9, 3.2, 4.9,7.2];					// approximations

function [a, b] = update_one_step(a, b, xx, yy, lambda)  	// lambda is learning rate
   rr = a*xx + b - yy;						// (minus) residuals
   nabla_a = rr*xx';						// gradient w.r.t. a
   nabla_b = rr*ones(xx)';					// gradient w.r.t. b
   // update
   a = a - lambda*nabla_a;					// gradient descent
   b = b - lambda*nabla_b;
endfunction

function pr_gr_ss = progress(aa,bb)
   delta_aa = max(aa) - min(aa);
   delta_bb = max(bb) - min(bb);
   pr_gr_ss = sqrt(delta_aa*delta_aa + delta_bb*delta_bb);
endfunction

function [a,b] = gradient_descent(xx,yy)
   aa = 1;							// initiate sequence of a's
   bb = 0;							// initiate sequence of b's
   lambda = .01;   						// learning rate
   epsilon = 10^-6; 						// progress tolerance
   // fill sequences aa and bb with 4 extra values
   for n = 1:4
      [a,b] = update_one_step(aa($), bb($), xx, yy, lambda);
      aa = [aa, a]; bb = [bb, b];
   end
   // continu if progress in not small enough
   while progress(aa,bb)>epsilon
      [a,b] = update_one_step(aa($), bb($), xx, yy, lambda);
      aa = [aa(2:$),a]; bb = [bb(2:$), b];
   end
   // return tail values
   a = aa($);
   b = bb($);
endfunction

// calculate and report
[a,b] = gradient_descent(xx,yy);
disp('a = '+string(a));						// a = 2.0599923
disp('b = '+string(b));						// b = -3.1599722

function y = f(x)
   y = a*x+b;
endfunction

clf()
CROSS = -2;
plot2d(xx,yy,CROSS);
x_min = min(xx); x_max = max(xx);
RED = 5;
plot2d([x_min,x_max],[f(x_min),f(x_max)],RED);
xtitle('best fitting line, using Gradient Descent');
xlabel('x'); ylabel('y');
