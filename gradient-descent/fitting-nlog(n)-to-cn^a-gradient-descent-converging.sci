// n.log(n)=O(n^(1+epsilon))
// f(n) = n.log(n)
// g(n) = n^a
// f(n) <= c.g(n)
//
// residue:
// r(n) = log(y(n)) - log(cn^a)
//      = log(y(n)) - log(c) - a.log(n)

nn = 10:10:100;
log_nn = log(nn);
yy = nn.*log_nn;
log_yy = log(yy);

function [a,c] = update_one_step(a, c, lambda) 		// lambda is learning rate
   rr = a*log_nn + log(c) - log_yy;			// (minus) residuals
   nabla_a = rr*log_nn';				// gradient w.r.t. a
   nabla_c = (c^-1)*rr*ones(nn)';			// gradient w.r.t. c
   // update
   a = a - lambda*nabla_a;				// gradient descent
   c = c - lambda*nabla_c;
endfunction

function [a,c] = update_thousand_steps(a, c, lambda)
   for i=1:1000
      [a,c] = update_one_step(a, c, lambda);
   end
endfunction

function pr_gr_ss = progress(aa,cc)
   delta_aa = max(aa) - min(aa);
   delta_cc = max(cc) - min(cc);
   pr_gr_ss = sqrt(delta_aa*delta_aa + delta_cc*delta_cc);
endfunction

function [a,c] = gradient_descent()
   aa = 1;						// initiate sequence of a's
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
disp('a = '+string(a));					// a = 1.2943467
disp('c = '+string(c));					// c = 1.2172339

function y = f(n)
   y = c*n.^a;
endfunction

plot2d(nn,yy,-1);					// crosses = -1
plot2d(10:100,f(10:100),5);				// red = 5
xtitle('fitting n.log(n) to c.n^a  i.e.  n.log(n)=O(n^a)')
xlabel('x')
ylabel('y')




