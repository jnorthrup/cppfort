
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf098-early-exit-pattern/cf098-early-exit-pattern_O2.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z16test_early_exitsiii>:
100000360: 37f80140    	tbnz	w0, #0x1f, 0x100000388 <__Z16test_early_exitsiii+0x28>
100000364: 37f80161    	tbnz	w1, #0x1f, 0x100000390 <__Z16test_early_exitsiii+0x30>
100000368: 37f80182    	tbnz	w2, #0x1f, 0x100000398 <__Z16test_early_exitsiii+0x38>
10000036c: 0b000028    	add	w8, w1, w0
100000370: 2a020108    	orr	w8, w8, w2
100000374: 1b007c29    	mul	w9, w1, w0
100000378: 1b027d29    	mul	w9, w9, w2
10000037c: 7100011f    	cmp	w8, #0x0
100000380: 1a8903e0    	csel	w0, wzr, w9, eq
100000384: d65f03c0    	ret
100000388: 12800000    	mov	w0, #-0x1               ; =-1
10000038c: d65f03c0    	ret
100000390: 12800020    	mov	w0, #-0x2               ; =-2
100000394: d65f03c0    	ret
100000398: 12800040    	mov	w0, #-0x3               ; =-3
10000039c: d65f03c0    	ret

00000001000003a0 <_main>:
1000003a0: 52800300    	mov	w0, #0x18               ; =24
1000003a4: d65f03c0    	ret
