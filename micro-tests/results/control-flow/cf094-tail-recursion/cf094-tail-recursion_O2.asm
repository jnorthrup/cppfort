
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf094-tail-recursion/cf094-tail-recursion_O2.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

00000001000003b0 <__Z16factorial_helperii>:
1000003b0: 7100081f    	cmp	w0, #0x2
1000003b4: 5400094b    	b.lt	0x1000004dc <__Z16factorial_helperii+0x12c>
1000003b8: 7100141f    	cmp	w0, #0x5
1000003bc: 54000863    	b.lo	0x1000004c8 <__Z16factorial_helperii+0x118>
1000003c0: 51000408    	sub	w8, w0, #0x1
1000003c4: 90000009    	adrp	x9, 0x100000000
1000003c8: 7100441f    	cmp	w0, #0x11
1000003cc: 54000082    	b.hs	0x1000003dc <__Z16factorial_helperii+0x2c>
1000003d0: 5280000a    	mov	w10, #0x0               ; =0
1000003d4: aa0003eb    	mov	x11, x0
1000003d8: 14000026    	b	0x100000470 <__Z16factorial_helperii+0xc0>
1000003dc: 121c6d0a    	and	w10, w8, #0xfffffff0
1000003e0: 4f000420    	movi.4s	v0, #0x1
1000003e4: 4f000421    	movi.4s	v1, #0x1
1000003e8: 4e041c21    	mov.s	v1[0], w1
1000003ec: 4e040c02    	dup.4s	v2, w0
1000003f0: 3dc18d23    	ldr	q3, [x9, #0x630]
1000003f4: 4ea38442    	add.4s	v2, v2, v3
1000003f8: 4b0a000b    	sub	w11, w0, w10
1000003fc: 6f000463    	mvni.4s	v3, #0x3
100000400: 6f0004e4    	mvni.4s	v4, #0x7
100000404: 6f000565    	mvni.4s	v5, #0xb
100000408: 6f0005e6    	mvni.4s	v6, #0xf
10000040c: aa0a03ec    	mov	x12, x10
100000410: 4f000427    	movi.4s	v7, #0x1
100000414: 4f000430    	movi.4s	v16, #0x1
100000418: 4ea38451    	add.4s	v17, v2, v3
10000041c: 4ea48452    	add.4s	v18, v2, v4
100000420: 4ea58453    	add.4s	v19, v2, v5
100000424: 4ea29c21    	mul.4s	v1, v1, v2
100000428: 4eb19c00    	mul.4s	v0, v0, v17
10000042c: 4eb29ce7    	mul.4s	v7, v7, v18
100000430: 4eb39e10    	mul.4s	v16, v16, v19
100000434: 4ea68442    	add.4s	v2, v2, v6
100000438: 7100418c    	subs	w12, w12, #0x10
10000043c: 54fffee1    	b.ne	0x100000418 <__Z16factorial_helperii+0x68>
100000440: 4ea19c00    	mul.4s	v0, v0, v1
100000444: 4ea09ce0    	mul.4s	v0, v7, v0
100000448: 4ea09e00    	mul.4s	v0, v16, v0
10000044c: 6e004001    	ext.16b	v1, v0, v0, #0x8
100000450: 0ea19c00    	mul.2s	v0, v0, v1
100000454: 1e26000c    	fmov	w12, s0
100000458: 0e0c3c0d    	mov.s	w13, v0[1]
10000045c: 1b0d7d81    	mul	w1, w12, w13
100000460: 6b0a011f    	cmp	w8, w10
100000464: 540003c0    	b.eq	0x1000004dc <__Z16factorial_helperii+0x12c>
100000468: 721e051f    	tst	w8, #0xc
10000046c: 540002c0    	b.eq	0x1000004c4 <__Z16factorial_helperii+0x114>
100000470: 121e750c    	and	w12, w8, #0xfffffffc
100000474: 4b0c0000    	sub	w0, w0, w12
100000478: 4f000420    	movi.4s	v0, #0x1
10000047c: 4e041c20    	mov.s	v0[0], w1
100000480: 4e040d61    	dup.4s	v1, w11
100000484: 3dc18d22    	ldr	q2, [x9, #0x630]
100000488: 4ea28421    	add.4s	v1, v1, v2
10000048c: 4b0c0149    	sub	w9, w10, w12
100000490: 6f000462    	mvni.4s	v2, #0x3
100000494: 4ea19c00    	mul.4s	v0, v0, v1
100000498: 4ea28421    	add.4s	v1, v1, v2
10000049c: 31001129    	adds	w9, w9, #0x4
1000004a0: 54ffffa1    	b.ne	0x100000494 <__Z16factorial_helperii+0xe4>
1000004a4: 6e004001    	ext.16b	v1, v0, v0, #0x8
1000004a8: 0ea19c00    	mul.2s	v0, v0, v1
1000004ac: 0e0c3c09    	mov.s	w9, v0[1]
1000004b0: 1e26000a    	fmov	w10, s0
1000004b4: 1b097d41    	mul	w1, w10, w9
1000004b8: 6b0c011f    	cmp	w8, w12
1000004bc: 54000061    	b.ne	0x1000004c8 <__Z16factorial_helperii+0x118>
1000004c0: 14000007    	b	0x1000004dc <__Z16factorial_helperii+0x12c>
1000004c4: aa0b03e0    	mov	x0, x11
1000004c8: 1b007c21    	mul	w1, w1, w0
1000004cc: 51000408    	sub	w8, w0, #0x1
1000004d0: 7100081f    	cmp	w0, #0x2
1000004d4: aa0803e0    	mov	x0, x8
1000004d8: 54ffff88    	b.hi	0x1000004c8 <__Z16factorial_helperii+0x118>
1000004dc: aa0103e0    	mov	x0, x1
1000004e0: d65f03c0    	ret

00000001000004e4 <__Z9factoriali>:
1000004e4: 7100081f    	cmp	w0, #0x2
1000004e8: 5400006a    	b.ge	0x1000004f4 <__Z9factoriali+0x10>
1000004ec: 52800020    	mov	w0, #0x1                ; =1
1000004f0: d65f03c0    	ret
1000004f4: 7100141f    	cmp	w0, #0x5
1000004f8: 54000062    	b.hs	0x100000504 <__Z9factoriali+0x20>
1000004fc: 52800028    	mov	w8, #0x1                ; =1
100000500: 14000043    	b	0x10000060c <__Z9factoriali+0x128>
100000504: 51000409    	sub	w9, w0, #0x1
100000508: 9000000a    	adrp	x10, 0x100000000
10000050c: 7100441f    	cmp	w0, #0x11
100000510: 540000a2    	b.hs	0x100000524 <__Z9factoriali+0x40>
100000514: 5280000b    	mov	w11, #0x0               ; =0
100000518: 52800028    	mov	w8, #0x1                ; =1
10000051c: aa0003ec    	mov	x12, x0
100000520: 14000025    	b	0x1000005b4 <__Z9factoriali+0xd0>
100000524: 4e040c00    	dup.4s	v0, w0
100000528: 3dc18d41    	ldr	q1, [x10, #0x630]
10000052c: 4ea18400    	add.4s	v0, v0, v1
100000530: 4f000421    	movi.4s	v1, #0x1
100000534: 121c6d2b    	and	w11, w9, #0xfffffff0
100000538: 6f000462    	mvni.4s	v2, #0x3
10000053c: 4b0b000c    	sub	w12, w0, w11
100000540: 6f0004e3    	mvni.4s	v3, #0x7
100000544: 6f000564    	mvni.4s	v4, #0xb
100000548: 6f0005e5    	mvni.4s	v5, #0xf
10000054c: aa0b03e8    	mov	x8, x11
100000550: 4f000426    	movi.4s	v6, #0x1
100000554: 4f000427    	movi.4s	v7, #0x1
100000558: 4f000430    	movi.4s	v16, #0x1
10000055c: 4ea28411    	add.4s	v17, v0, v2
100000560: 4ea38412    	add.4s	v18, v0, v3
100000564: 4ea48413    	add.4s	v19, v0, v4
100000568: 4ea19c01    	mul.4s	v1, v0, v1
10000056c: 4ea69e26    	mul.4s	v6, v17, v6
100000570: 4ea79e47    	mul.4s	v7, v18, v7
100000574: 4eb09e70    	mul.4s	v16, v19, v16
100000578: 4ea58400    	add.4s	v0, v0, v5
10000057c: 71004108    	subs	w8, w8, #0x10
100000580: 54fffee1    	b.ne	0x10000055c <__Z9factoriali+0x78>
100000584: 4ea19cc0    	mul.4s	v0, v6, v1
100000588: 4ea09ce0    	mul.4s	v0, v7, v0
10000058c: 4ea09e00    	mul.4s	v0, v16, v0
100000590: 6e004001    	ext.16b	v1, v0, v0, #0x8
100000594: 0ea19c00    	mul.2s	v0, v0, v1
100000598: 1e260008    	fmov	w8, s0
10000059c: 0e0c3c0d    	mov.s	w13, v0[1]
1000005a0: 1b0d7d08    	mul	w8, w8, w13
1000005a4: 6b0b013f    	cmp	w9, w11
1000005a8: 540003c0    	b.eq	0x100000620 <__Z9factoriali+0x13c>
1000005ac: 721e053f    	tst	w9, #0xc
1000005b0: 540002c0    	b.eq	0x100000608 <__Z9factoriali+0x124>
1000005b4: 121e752d    	and	w13, w9, #0xfffffffc
1000005b8: 4b0d0000    	sub	w0, w0, w13
1000005bc: 4f000420    	movi.4s	v0, #0x1
1000005c0: 4e041d00    	mov.s	v0[0], w8
1000005c4: 4e040d81    	dup.4s	v1, w12
1000005c8: 3dc18d42    	ldr	q2, [x10, #0x630]
1000005cc: 4ea28421    	add.4s	v1, v1, v2
1000005d0: 4b0d0168    	sub	w8, w11, w13
1000005d4: 6f000462    	mvni.4s	v2, #0x3
1000005d8: 4ea09c20    	mul.4s	v0, v1, v0
1000005dc: 4ea28421    	add.4s	v1, v1, v2
1000005e0: 31001108    	adds	w8, w8, #0x4
1000005e4: 54ffffa1    	b.ne	0x1000005d8 <__Z9factoriali+0xf4>
1000005e8: 6e004001    	ext.16b	v1, v0, v0, #0x8
1000005ec: 0ea19c00    	mul.2s	v0, v0, v1
1000005f0: 0e0c3c08    	mov.s	w8, v0[1]
1000005f4: 1e26000a    	fmov	w10, s0
1000005f8: 1b087d48    	mul	w8, w10, w8
1000005fc: 6b0d013f    	cmp	w9, w13
100000600: 54000061    	b.ne	0x10000060c <__Z9factoriali+0x128>
100000604: 14000007    	b	0x100000620 <__Z9factoriali+0x13c>
100000608: aa0c03e0    	mov	x0, x12
10000060c: 1b087c08    	mul	w8, w0, w8
100000610: 51000409    	sub	w9, w0, #0x1
100000614: 7100081f    	cmp	w0, #0x2
100000618: aa0903e0    	mov	x0, x9
10000061c: 54ffff88    	b.hi	0x10000060c <__Z9factoriali+0x128>
100000620: aa0803e0    	mov	x0, x8
100000624: d65f03c0    	ret

0000000100000628 <_main>:
100000628: 52800f00    	mov	w0, #0x78               ; =120
10000062c: d65f03c0    	ret
