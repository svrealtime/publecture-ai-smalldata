// Standard Gaussian
function y = phi(z)
   y = exp(-z.*z/2)/sqrt(2*%pi);
endfunction

xx = -5:5;
xx2 = xx.*xx;
yy = phi(xx);
log_yy = log(yy);

function [a,c] = update_one_step(a, c, lambda) 		// lambda is learning rate
   rr = a*xx2 + log(c) - log_yy;			// (minus) residuals
   nabla_a = rr*xx2';					// gradient w.r.t. a
   nabla_c = (c^-1)*rr*ones(xx)';			// gradient w.r.t. c
   // update
   a = a - lambda*nabla_a;				// gradient descent
   c = c - lambda*nabla_c;
endfunction

function pr_gr_ss = progress(aa,cc)
   delta_aa = max(aa) - min(aa);
   delta_cc = max(cc) - min(cc);
   pr_gr_ss = sqrt(delta_aa*delta_aa + delta_cc*delta_cc);
endfunction

function [a,c] = gradient_descent()
   aa = -1;						// initiate sequence of a's
   cc = 1;						// initiate sequence of b's
   lambda = .001;   					// learning rate
   epsilon = 10^-6; 					// progress tolerance
   // fill sequences aa and cc with 4 extra values
   for n = 1:4
      [a,c] = update_one_step(aa($), cc($), lambda);
      aa = [aa, a]; cc = [cc, c];
   end
   // continu if progress in not small enough
   while progress(aa,cc)>epsilon
      [a,c] = update_one_step(aa($), cc($), lambda);
      aa = [aa(2:$),a]; cc = [cc(2:$), c];
   end
   // return tail values
   a = aa($);
   c = cc($);
endfunction

// calculate and report
[a,c] = gradient_descent();
disp('a = '+string(a));					// a = -0.4999995
disp('c = '+string(c));					// c = 0.3989424

function y = f(x)
   y = c*exp(a*x.*x);
endfunction

plot2d(xx,yy,-1);					// crosses = -1
plot2d(-5:.1:5,f(-5:.1:5),5);				// red = 5




