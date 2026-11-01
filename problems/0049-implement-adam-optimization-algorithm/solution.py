import numpy as np

def adam_optimizer(f, grad, x0, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, num_iterations=10):
	# Your code here
	m_t=torch.zeros_like(x0)
	v_t=torch.zeros_like(x0)
	t=0
	for _ in range(num_iterations):
		t=t+1
		g_t=grad(x0)
		m_t=beta1*(m_t)+(1-beta1)*g_t
		v_t=beta2*(v_t)+(1-beta2)*(torch.square(g_t))
		m_t_=(m_t)/(1-beta1**t)
		v_t_=(v_t)/(1-beta2**t)
		x0-=learning_rate*((m_t_)/(torch.sqrt(v_t_)+epsilon))
	return x0
