
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf094-tail-recursion/cf094-tail-recursion_O1.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z16factorial_helperii>:
100000360: 7100081f    	cmp	w0, #0x2
100000364: 540000ab    	b.lt	0x100000378 <__Z16factorial_helperii+0x18>
100000368: 1b007c21    	mul	w1, w1, w0
10000036c: 51000400    	sub	w0, w0, #0x1
100000370: 7100081f    	cmp	w0, #0x2
100000374: 54ffffaa    	b.ge	0x100000368 <__Z16factorial_helperii+0x8>
100000378: aa0103e0    	mov	x0, x1
10000037c: d65f03c0    	ret

0000000100000380 <__Z9factoriali>:
100000380: 52800021    	mov	w1, #0x1                ; =1
100000384: 17fffff7    	b	0x100000360 <__Z16factorial_helperii>

0000000100000388 <_main>:
100000388: 528000a0    	mov	w0, #0x5                ; =5
10000038c: 52800021    	mov	w1, #0x1                ; =1
100000390: 17fffff4    	b	0x100000360 <__Z16factorial_helperii>
