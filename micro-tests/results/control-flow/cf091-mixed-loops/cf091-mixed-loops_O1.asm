
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf091-mixed-loops/cf091-mixed-loops_O1.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z16test_mixed_loopsv>:
100000360: 52800008    	mov	w8, #0x0                ; =0
100000364: 52800009    	mov	w9, #0x0                ; =0
100000368: 52800000    	mov	w0, #0x0                ; =0
10000036c: 1280002a    	mov	w10, #-0x2              ; =-2
100000370: 1100054b    	add	w11, w10, #0x1
100000374: 1100094c    	add	w12, w10, #0x2
100000378: 9baa7d6a    	umull	x10, w11, w10
10000037c: d341fd4a    	lsr	x10, x10, #1
100000380: 0b08000d    	add	w13, w0, w8
100000384: 1b0a358a    	madd	w10, w12, w10, w13
100000388: 7100013f    	cmp	w9, #0x0
10000038c: 1a8a0000    	csel	w0, w0, w10, eq
100000390: 0b090108    	add	w8, w8, w9
100000394: 11000929    	add	w9, w9, #0x2
100000398: aa0b03ea    	mov	x10, x11
10000039c: 71000d7f    	cmp	w11, #0x3
1000003a0: 54fffe81    	b.ne	0x100000370 <__Z16test_mixed_loopsv+0x10>
1000003a4: d65f03c0    	ret

00000001000003a8 <_main>:
1000003a8: 52800460    	mov	w0, #0x23               ; =35
1000003ac: d65f03c0    	ret
