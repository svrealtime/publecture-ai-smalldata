// n.log(n)=O(n^(1+epsilon))
// f(n) = n.log(n)
// g(n) = n^a
// f(n) <= c.g(n)

nn = 10:10:100;
yy = nn.*log(nn);

function [a,c] = update_one_step(a, c, lambda) 	// lambda is learning rate
   rr = c*nn.^a - yy;				// (minus) residuals
   nabla_a = a*c*rr*(nn.^(a-1))';		// gradient w.r.t. a
   nabla_c = rr*(nn.^a)';			// gradient w.r.t. c
   // update
   a = a - lambda*nabla_a;			// gradient descent
   c = c - lambda*nabla_c;
endfunction

function [a,c] = update_thousand_steps(a, c, lambda)
   for i=1:1000
      [a,c] = update_one_step(a, c, lambda);
   end
endfunction
