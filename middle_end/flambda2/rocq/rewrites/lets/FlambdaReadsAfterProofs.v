(******************************************************************************
 *                                  OxCaml                                    *
 * -------------------------------------------------------------------------- *
 *                               MIT License                                  *
 *                                                                            *
 * Copyright (c) 2026 Jane Street Group LLC                                   *
 * opensource-contacts@janestreet.com                                         *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included    *
 * in all copies or substantial portions of the Software.                     *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER *
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING    *
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER        *
 * DEALINGS IN THE SOFTWARE.                                                  *
 ******************************************************************************)

(* Proofs of the statements of FlambdaReadsAfterSpec.v. Nothing here
   needs review. What it means for what is known of the blocks to hold,
   and that it stays so, are those of the pass of the term before
   (FlambdaBlockLoadProofs.v). The removal follows
   FlambdaUnusedProjectionsProofs.v. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaKindModel FlambdaValueSpec FlambdaPrimitiveModel
  FlambdaPrimitiveSpec FlambdaPrimitiveProofs FlambdaSemanticsSpec
  FlambdaSemanticsProofs FlambdaContextSpec FlambdaContextProofs
  FlambdaNominalSpec FlambdaNominalProofs FlambdaNormaliseSpec
  FlambdaNormaliseProofs FlambdaValidatorEquivarianceBaseProofs
  FlambdaBlockLoadSpec FlambdaBlockLoadProofs FlambdaReadsAfterSpec
  FlambdaEquivarianceTactic.
Import ListNotations.

(* ======================================================================== *)
(* What a simple is, by its form                                            *)
(* ======================================================================== *)

Lemma surely_value_kind (vars : env) (s : simple) (v : value) :
  surely_value s = true ->
  eval_simple vars s = Some v ->
  kind_eqb (kind_of_value v) Value = true.
Proof.
  destruct s as [x | sym | c]; simpl; try discriminate.
  - intros _ [= <-]. reflexivity.
  - destruct c; try discriminate. intros _ [= <-]. reflexivity.
Qed.

Lemma surely_immediate_tagged (vars : env) (s : simple) (v : value) :
  surely_immediate s = true ->
  eval_simple vars s = Some v ->
  is_tagged_immediate v = true.
Proof.
  destruct s as [x | sym | c]; simpl; try discriminate.
  destruct c; try discriminate. intros Hrange [= <-].
  unfold is_tagged_immediate. simpl. rewrite Hrange. reflexivity.
Qed.

(* ======================================================================== *)
(* Reading a known block                                                    *)
(* ======================================================================== *)

(* Where the pass reads, the primitive has a value, the one read; and
   elsewhere the defining expression is as it was. *)
Lemma load_known_named_surely_exact (known : known_blocks) (vars : env)
    (statics : static_env) (d : named) :
  blocks_hold known vars statics ->
  eval_named statics vars (load_known_named_surely known d)
  = eval_named statics vars d.
Proof.
  intros Hhold. unfold load_known_named_surely.
  destruct d as [s | p args | code slots | fslot decls slots | slot closure
    | pfrom slot closure | move_from move_to closure];
    try reflexivity.
  destruct (unary_applied p args) as [[u block] | ] eqn:Happlied;
    [ | reflexivity ].
  destruct (known_block_of known block) as [known_block | ] eqn:Hknown;
    [ | reflexivity ].
  destruct (of_known_block_surely u known_block) as [result | ] eqn:Hresult;
    [ | reflexivity ].
  destruct (unary_applied_spec _ _ _ _ Happlied) as [-> ->].
  destruct (known_block_of_sound _ _ _ _ _ Hhold Hknown)
    as (vs & block_value & Hfields & Hblock & Hblock_of & Himmediate).
  simpl. rewrite Hblock. simpl. rewrite eval_prim_unary.
  unfold of_known_block_surely in Hresult.
  destruct u; try discriminate Hresult.
  - (* [Block_load] *)
    destruct (of_known_block (Block_load kind mut field) known_block)
      as [contents | ] eqn:Hof; [ | discriminate Hresult ].
    destruct (surely_value contents) eqn:Hsure; [ | discriminate Hresult ].
    injection Hresult as <-.
    assert (Hfield : field_loaded kind field known_block = Some contents)
      by (simpl in Hof; destruct mut; [ discriminate Hof | exact Hof .. ]).
    destruct (field_loaded_spec _ _ _ _ Hfield)
      as (tag & size & -> & Htag & Hsize & Hpos & Hnth).
    destruct (fields_hold_nth _ _ _ Hfields _ _ Hnth)
      as (loaded & Hloaded & Hcontents).
    simpl. rewrite Hblock_of. simpl.
    rewrite Htag, (fields_hold_length _ _ _ Hfields), Hsize, Hpos. simpl.
    rewrite Hloaded. simpl.
    rewrite (surely_value_kind vars contents loaded Hsure Hcontents).
    exact Hcontents.
  - (* [Is_int] *)
    simpl in Hresult. injection Hresult as <-. simpl.
    unfold eval_is_int. rewrite Himmediate.
    destruct variant_only.
    + rewrite Hblock_of. reflexivity.
    + unfold is_allocated. rewrite Hblock_of. simpl.
      rewrite orb_true_r. reflexivity.
  - (* [Get_tag] *)
    simpl in Hresult. injection Hresult as <-. simpl.
    rewrite Hblock_of. reflexivity.
Qed.

(* The machine runs a [Let] with the heap ([exec_named]); the two
   lemmas above are about the value alone ([eval_named]). The
   primitives that the pass replaces may read a cell; but the block is
   known, so its value is a block that a primitive made or a symbol
   bound to one, and no location: the run of the primitive has the
   value that [eval_named] gives and leaves the heap and the record
   (FlambdaSemanticsSpec.exec_named_no_location_statement), and so does
   the simple that replaces it. *)
Lemma load_known_named_surely_exec (known : known_blocks) (vars : env)
    (statics : static_env) (hp : heap) (rd : read_record) (d : named) :
  blocks_hold known vars statics ->
  exec_named statics hp rd vars (load_known_named_surely known d)
  = exec_named statics hp rd vars d.
Proof.
  intros Hhold.
  generalize (load_known_named_surely_exact known vars statics d Hhold).
  unfold load_known_named_surely.
  destruct d as [s | p args | code slots | fslot decls slots | slot closure
    | pfrom slot closure | move_from move_to closure];
    try (intros _; reflexivity).
  destruct (unary_applied p args) as [[u block] | ] eqn:Happlied;
    [ | intros _; reflexivity ].
  destruct (known_block_of known block) as [known_block | ] eqn:Hknown;
    [ | intros _; reflexivity ].
  destruct (of_known_block_surely u known_block) as [result | ]
    eqn:Hresult; [ | intros _; reflexivity ].
  destruct (unary_applied_spec _ _ _ _ Happlied) as [-> ->].
  intros Hexact.
  destruct (known_block_of_sound _ _ _ _ _ Hhold Hknown)
    as (vs & block_value & _ & Hblock & Hblock_of & _).
  rewrite
    (exec_named_no_location statics hp rd vars (Unary u) [block]
       [block_value]).
  - cbn [exec_named]. rewrite Hexact. reflexivity.
  - intros q. discriminate.
  - cbn [eval_simples]. rewrite Hblock. reflexivity.
  - intros l [-> | []]. discriminate Hblock_of.
Qed.

Fixpoint load_blocks_knowing_surely_equivalent (e : expr) :
  ∀ (known : known_blocks) (vars : env) (conts : cont_env)
    (statics : static_env) (hp : heap) (rd : read_record),
    blocks_hold known vars statics ->
    equivalent (State e vars conts statics hp rd)
      (State (load_blocks_knowing_surely known e) vars conts statics hp rd)
with load_blocks_in_handler_surely_equivalent (h : continuation_handler) :
  ∀ (known : known_blocks) (k : continuation) (body : expr) (vars : env)
    (conts : cont_env) (statics : static_env) (hp : heap) (rd : read_record),
    blocks_hold known vars statics ->
    equivalent
      (State (Let_cont_non_recursive k h body) vars conts statics hp rd)
      (State
         (Let_cont_non_recursive k (load_blocks_in_handler_surely known h) body)
         vars conts statics hp rd).
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap
         | ac | scrutinee arms | ];
      intros known vars conts statics hp rd Hhold; simpl.
    + (* Let: the machine runs [d] with the heap; what the pass puts
         for it runs the same. *)
      apply let_equivalent_step_exec.
      * exact (load_known_named_surely_exec known vars statics hp rd d Hhold).
      * intros v hp' rd' Hd _. apply load_blocks_knowing_surely_equivalent.
        exact
          (blocks_hold_under_let_exec known x d vars statics hp hp' rd rd' v
             Hhold Hd).
    + (* Let_static *)
      apply let_static_equivalent_step.
      destruct (eval_static_bindings vars bindings statics)
        as [extended | ] eqn:Hbindings; [ | exact I ].
      apply load_blocks_knowing_surely_equivalent.
      exact (blocks_hold_under_statics known vars statics bindings extended
        Hhold Hbindings).
    + (* Let_cont_non_recursive *)
      apply (equivalent_trans _
        (State
           (Let_cont_non_recursive k
              (load_blocks_in_handler_surely known h) body)
           vars conts statics hp rd)).
      * apply load_blocks_in_handler_surely_equivalent, Hhold.
      * apply let_cont_non_recursive_equivalent_step,
          load_blocks_knowing_surely_equivalent, Hhold.
    + (* Let_cont_recursive *)
      apply let_cont_recursive_equivalent_step; [ reflexivity | intros _ ].
      apply load_blocks_knowing_surely_equivalent, Hhold.
    + apply equivalent_refl.
    + apply equivalent_refl.
    + apply equivalent_refl.
    + apply equivalent_refl.
  - destruct h as [params metadata handler].
    intros known k body vars conts statics hp rd Hhold. simpl.
    apply refines_antisymmetric;
      apply (let_cont_handler_refines_where (symbols_hold known vars)
        (symbols_hold_extend known vars));
      try exact (proj2 Hhold);
      intros vs handler_vars conts' statics' hp' rd' Hsymbols _ Hbind;
      apply equivalent_refines; [ | apply equivalent_sym ];
      apply load_blocks_knowing_surely_equivalent;
      (apply (blocks_hold_bind_params statics' params known vs vars
         handler_vars);
       [ split; [ exact (proj1 Hhold) | exact Hsymbols ] | exact Hbind ]).
Qed.

Theorem load_known_blocks_surely_equivalent :
  load_known_blocks_surely_equivalent_statement.
Proof.
  intros e vars conts statics hp rd. apply
    load_blocks_knowing_surely_equivalent.
  split; intros entry [].
Qed.

Theorem load_known_named_surely_loads : load_known_named_surely_loads_statement.
Proof.
  intros known d. unfold load_known_named_surely, load_known_named.
  destruct d as [s | p args | code slots | fslot decls slots | slot closure
    | pfrom slot closure | move_from move_to closure];
    try (left; reflexivity).
  destruct (unary_applied p args) as [[u block] | ]; [ | left; reflexivity ].
  destruct (known_block_of known block) as [known_block | ];
    [ | left; reflexivity ].
  unfold of_known_block_surely.
  destruct u; try (left; reflexivity); try (right; reflexivity).
  destruct (of_known_block (Block_load kind mut field) known_block)
    as [contents | ]; [ | left; reflexivity ].
  destruct (surely_value contents); [ right | left ]; reflexivity.
Qed.

(* ======================================================================== *)
(* Removing an allocation of constants that nothing uses                    *)
(* ======================================================================== *)

(* Constant tagged immediates in range have values, which are tagged
   immediates. *)
Lemma constants_evaluated (vars : env) (args : list simple) :
  forallb surely_immediate args = true ->
  ∃ vs : list value,
    eval_simples vars args = Some vs
    ∧ forallb is_tagged_immediate vs = true.
Proof.
  induction args as [ | s args IH]; simpl.
  - intros _. exists []. split; reflexivity.
  - intros Hargs. apply andb_true_iff in Hargs as [Hs Hargs].
    destruct (IH Hargs) as (vs & Heval & Htagged).
    destruct s as [x | sym | c]; try discriminate Hs.
    exists (Value_const c :: vs). simpl. rewrite Heval. simpl.
    split; [ reflexivity | ].
    rewrite Htagged, andb_true_r.
    exact (surely_immediate_tagged vars (Const c) _ Hs eq_refl).
Qed.

(* A tagged immediate is a value of kind [Value] and no symbol. *)
Lemma tagged_immediate_kind (v : value) :
  is_tagged_immediate v = true ->
  kind_eqb (kind_of_value v) Value = true
  ∧ ∀ statics : static_env, bound_if_symbol statics v = true.
Proof.
  unfold is_tagged_immediate.
  destruct v as [c | | | | | | | | ]; simpl; try discriminate.
  destruct c; simpl; try discriminate.
  intros _. split; [ | intros statics ]; reflexivity.
Qed.

Theorem allocation_of_constants_defined :
  allocation_of_constants_defined_statement.
Proof.
  intros statics vars d Hd.
  destruct d as [s | p args | code slots | fslot decls slots | slot closure
    | pfrom slot closure | move_from move_to closure];
    try discriminate Hd.
  assert (Hp : ∃ vp : variadic_primitive, p = Variadic vp)
    by (destruct p; try discriminate Hd; eexists; reflexivity).
  destruct Hp as [vp ->].
  assert (Hargs : forallb surely_immediate args = true)
    by (destruct vp as [[tag] | kind]; exact Hd).
  destruct (constants_evaluated vars args Hargs) as (vs & Heval & Htagged).
  assert
    (Hvalues :
      forallb (fun v => kind_eqb (kind_of_value v) Value) vs = true
      ∧ forallb (bound_if_symbol statics) vs = true).
  { clear Heval. induction vs as [ | v vs IH]; [ split; reflexivity | ].
    simpl in Htagged. apply andb_true_iff in Htagged as [Hv Hvs].
    destruct (IH Hvs) as [Hkinds Hbound].
    destruct (tagged_immediate_kind v Hv) as [Hkind Hsymbol].
    simpl. rewrite Hkind, Hkinds, (Hsymbol statics), Hbound.
    split; reflexivity. }
  destruct Hvalues as [Hkinds Hbound].
  simpl. rewrite Heval. simpl. rewrite eval_prim_variadic.
  destruct vp as [[tag] | kind]; simpl.
  - rewrite Hkinds, Hbound. simpl.
    exists (Value_block tag vs). split; reflexivity.
  - assert (Hmade : array_made kind vs = Some vs).
    { unfold array_made.
      assert (Hok : forallb (array_element_ok kind) vs = true).
      { clear Heval Hbound.
        induction vs as [ | v vs IH]; [ reflexivity | ].
        simpl in Htagged, Hkinds |- *.
        apply andb_true_iff in Htagged as [Hv Hvs].
        apply andb_true_iff in Hkinds as [Hkind Hkinds].
        rewrite (IH Hvs Hkinds), andb_true_r.
        destruct kind; simpl; [ exact Hv | exact Hkind .. ]. }
      rewrite Hok. reflexivity. }
    rewrite Hmade. simpl. rewrite Hbound.
    exists (Value_immutable_array vs). split; reflexivity.
Qed.

(* The allocation has a value of the kind of the [Let], so the [Let]
   takes its step; the body does not look at the variable. *)
Lemma unused_constant_allocation_let (x : variable) (kind : flambda_kind)
    (d : named) (body : expr) :
  kind_eqb kind Value = true ->
  allocation_of_constants d = true ->
  ~ var_free_in_expr x body ->
  expr_equivalent (Let x kind d body) body.
Proof.
  intros Hkind Hdefined Hunused vars conts statics hp rd.
  destruct (allocation_of_constants_defined statics vars d Hdefined)
    as (v & Hd & Hv).
  apply kind_eqb_eq in Hkind. subst kind.
  assert (Hkind : kind_eqb (kind_of_value v) Value = true)
    by (apply kind_eqb_eq, Hv).
  (* The allocation is no primitive of the heap and its arguments are
     constants: the run has the value and leaves the heap and the
     record. *)
  assert (Hrun : exec_named statics hp rd vars d = Some (v, hp, rd)).
  { destruct d as [s | p args | code slots | fslot decls slots
      | slot closure | pfrom slot closure | move_from move_to closure];
      try discriminate Hdefined.
    assert (Hargs : forallb surely_immediate args = true).
    { destruct p as [u | b | vp | hprim | rp]; try discriminate Hdefined.
      destruct vp as [[tag] | akind]; exact Hdefined. }
    destruct (constants_evaluated vars args Hargs) as (vs & Hvs & Htagged).
    rewrite (exec_named_no_location statics hp rd vars p args vs).
    - rewrite Hd. reflexivity.
    - intros q ->. discriminate Hdefined.
    - exact Hvs.
    - intros l Hl.
      pose proof (proj1 (forallb_forall _ _) Htagged _ Hl) as Himmediate.
      discriminate Himmediate. }
  apply (equivalent_trans _
    (State body ((x, v) :: vars) conts statics hp rd));
    [ apply step_equivalent; rewrite step_let_exec, Hrun, Hkind;
      reflexivity | ].
  apply coincide_equivalent.
  - intros y Hy. simpl.
    destruct (Pos.eqb_spec y x) as [-> | _]; [ contradiction | reflexivity ].
  - intros k _. apply (conts_agree_same identity_renaming). reflexivity.
Qed.

Lemma let_cont_handler_equivalent (k : continuation)
    (params : list variable) (metadata : handler_metadata)
    (handler handler' body : expr) :
  expr_equivalent handler handler' ->
  expr_equivalent
    (Let_cont_non_recursive k (Handler params metadata handler) body)
    (Let_cont_non_recursive k (Handler params metadata handler') body).
Proof.
  intros Hhandler vars conts statics hp rd.
  apply refines_antisymmetric; apply let_cont_handler_refines_at;
    intros values handler_vars conts' statics' hp' rd' _ Hbind;
    apply equivalent_refines; [ | apply equivalent_sym ];
    apply Hhandler.
Qed.

Fixpoint remove_constant_allocations_equivalent_aux (e : expr) :
  expr_equivalent e (remove_constant_allocations e).
Proof.
  destruct e
    as [ x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    simpl;
    try (intros vars conts statics hp rd; apply equivalent_refl).
  - assert
      (Hbody :
        expr_equivalent (Let x kind d body)
          (Let x kind d (remove_constant_allocations body))).
    { intros vars conts statics hp rd.
      apply let_equivalent_step_exec; [ reflexivity | intros v hp' rd' _ _ ].
      apply remove_constant_allocations_equivalent_aux. }
    destruct
      (kind_eqb kind Value
       && allocation_of_constants d
       && negb (var_free_b x (remove_constant_allocations body)))
      eqn:Hremove; [ | exact Hbody ].
    apply andb_true_iff in Hremove as [Hdefined Hunused].
    apply andb_true_iff in Hdefined as [Hkind Hdefined].
    apply negb_true_iff in Hunused.
    apply (expr_equivalent_trans _ _ _ Hbody),
      unused_constant_allocation_let; [ exact Hkind | exact Hdefined | ].
    intros Hx. apply var_free_b_sound in Hx. congruence.
  - intros vars conts statics hp rd.
    apply let_static_equivalent_step.
    destruct (eval_static_bindings vars bindings statics) as [extended | ];
      [ | exact I ].
    apply remove_constant_allocations_equivalent_aux.
  - apply (expr_equivalent_trans _
      (Let_cont_non_recursive k
         (Handler params metadata (remove_constant_allocations handler))
         body)).
    + apply let_cont_handler_equivalent,
        remove_constant_allocations_equivalent_aux.
    + intros vars conts statics hp rd.
      apply let_cont_non_recursive_equivalent_step.
      apply remove_constant_allocations_equivalent_aux.
  - intros vars conts statics hp rd.
    apply let_cont_recursive_equivalent_step; [ reflexivity | intros _ ].
    apply remove_constant_allocations_equivalent_aux.
Qed.

Theorem remove_constant_allocations_equivalent :
  remove_constant_allocations_equivalent_statement.
Proof. exact remove_constant_allocations_equivalent_aux. Qed.

(* ======================================================================== *)
(* Renaming                                                                 *)
(* ======================================================================== *)

(* The laws are proved by the tactic of FlambdaEquivarianceTactic.v from
   the definitions; it states and proves by itself the laws of
   [surely_immediate] and of [allocation_of_constants]. The law of
   [var_free_b] is given to it, which it would otherwise derive again.
   [surely_value_renamed] is used by name in
   FlambdaValidatorEquivarianceProofs.v. *)
Lemma surely_value_renamed (r : renaming) (s : simple) :
  surely_value (rename_simple r s) = surely_value s.
Proof. equivariance. Qed.

#[local] Hint Rewrite var_free_b_equivariant_aux : equivariance.

Theorem remove_constant_allocations_equivariant :
  remove_constant_allocations_equivariant_statement.
Proof. equivariance. Qed.
