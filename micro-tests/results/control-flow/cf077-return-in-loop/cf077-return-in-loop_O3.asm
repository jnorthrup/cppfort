
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf077-return-in-loop/cf077-return-in-loop_O3.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z19test_return_in_loopi>:
100000360: aa0003e8    	mov	x8, x0
100000364: 52800c69    	mov	w9, #0x63               ; =99
100000368: 71018c1f    	cmp	w0, #0x63
10000036c: 1a893000    	csel	w0, w0, w9, lo
100000370: 7100111f    	cmp	w8, #0x4
100000374: 54000062    	b.hs	0x100000380 <__Z19test_return_in_loopi+0x20>
100000378: 52800009    	mov	w9, #0x0                ; =0
10000037c: 14000009    	b	0x1000003a0 <__Z19test_return_in_loopi+0x40>
100000380: 11000409    	add	w9, w0, #0x1
100000384: 7200052a    	ands	w10, w9, #0x3
100000388: 5280008b    	mov	w11, #0x4               ; =4
10000038c: 1a8a016a    	csel	w10, w11, w10, eq
100000390: 4b0a0129    	sub	w9, w9, w10
100000394: aa0903ea    	mov	x10, x9
100000398: 7100114a    	subs	w10, w10, #0x4
10000039c: 54ffffe1    	b.ne	0x100000398 <__Z19test_return_in_loopi+0x38>
1000003a0: 4b090108    	sub	w8, w8, w9
1000003a4: 51019129    	sub	w9, w9, #0x64
1000003a8: 340000a8    	cbz	w8, 0x1000003bc <__Z19test_return_in_loopi+0x5c>
1000003ac: 51000508    	sub	w8, w8, #0x1
1000003b0: 31000529    	adds	w9, w9, #0x1
1000003b4: 54ffffa3    	b.lo	0x1000003a8 <__Z19test_return_in_loopi+0x48>
1000003b8: 12800000    	mov	w0, #-0x1               ; =-1
1000003bc: d65f03c0    	ret

00000001000003c0 <_main>:
1000003c0: 52800540    	mov	w0, #0x2a               ; =42
1000003c4: d65f03c0    	ret
