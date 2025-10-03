
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf084-short-circuit-bounds-check/cf084-short-circuit-bounds-check_O3.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z17test_bounds_checkPiii>:
100000360: 37f800e2    	tbnz	w2, #0x1f, 0x10000037c <__Z17test_bounds_checkPiii+0x1c>
100000364: 6b01005f    	cmp	w2, w1
100000368: 540000aa    	b.ge	0x10000037c <__Z17test_bounds_checkPiii+0x1c>
10000036c: b8625800    	ldr	w0, [x0, w2, uxtw #2]
100000370: 7100001f    	cmp	w0, #0x0
100000374: 5400004d    	b.le	0x10000037c <__Z17test_bounds_checkPiii+0x1c>
100000378: d65f03c0    	ret
10000037c: 12800000    	mov	w0, #-0x1               ; =-1
100000380: d65f03c0    	ret

0000000100000384 <_main>:
100000384: 52800060    	mov	w0, #0x3                ; =3
100000388: d65f03c0    	ret
