
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf064-goto-cleanup-pattern/cf064-goto-cleanup-pattern_O2.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z17test_goto_cleanupi>:
100000360: 7100001f    	cmp	w0, #0x0
100000364: 52800548    	mov	w8, #0x2a               ; =42
100000368: 5a9f1108    	csinv	w8, w8, wzr, ne
10000036c: 5a9fa100    	csinv	w0, w8, wzr, ge
100000370: d65f03c0    	ret

0000000100000374 <_main>:
100000374: 52800540    	mov	w0, #0x2a               ; =42
100000378: d65f03c0    	ret
