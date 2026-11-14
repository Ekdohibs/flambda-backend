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

(* Proofs of the statements of FlambdaMergeTablesSpec.v. Nothing here
   needs review.

   The refinement of a [Switch] is by the steps of the machine, as in
   FlambdaMergeSwitchesProofs.v: where the [Switch] has a step it goes
   to an arm, which the test has found to be the merged application
   with an element of the table in place of the variable; the merged
   term binds the variable to the value of that element in the steps
   of its [Let]s, the last of which is the load, and then the two
   apply the same continuation to the same values. With a way for
   each argument it goes as
   FlambdaMergeSwitchesProofs.merge_switch_by_arguments_refines: the
   [Let]s of a way leave its name bound to the value that the
   argument has in the arm, those of a table by the same steps as
   above, and the names bound after it change no argument that is
   the same in every arm. The load has that value because the table
   is bound to the values of its elements ([tables_bound]), which a
   group of static bindings that has a value makes true of its own
   tables and keeps true of the others
   ([tables_under_statics_bound]). Down a term and in the bodies of
   code, the recursion is that of
   FlambdaUnboxInScopeProofs.rewrite_code_in_scope_refines_aux, with
   the tables for what is known of symbols. *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaValueSpec FlambdaPrimitiveModel FlambdaPrimitiveSpec
  FlambdaPrimitiveProofs FlambdaSemanticsSpec FlambdaSemanticsProofs
  FlambdaWellFormedSpec FlambdaWellFormedProofs FlambdaContextSpec
  FlambdaContextProofs FlambdaNominalSpec FlambdaNominalProofs
  FlambdaAlphaProofs FlambdaNormaliseSpec FlambdaNormaliseProofs
  FlambdaMergeSwitchesSpec FlambdaMergeSwitchesProofs FlambdaMergeTablesSpec.
Import ListNotations.

Local Open Scope Z_scope.

(* ======================================================================== *)
(* The values of the merged arguments                                       *)
(* ======================================================================== *)

(* Under a binding of the variable to the value of an element, an
   argument has the value that it has with the element in place of the
   variable, in any environment that binds the other variables alike. *)
Lemma eval_simple_element_for (v : variable) (element : simple) (w : value)
    (vars vars' : env) (s : simple) :
  eval_simple [] element = Some w ->
  (∀ x : variable, x <> v -> lookup x vars' = lookup x vars) ->
  eval_simple ((v, w) :: vars') s = eval_simple vars (element_for v element s).
Proof.
  intros Helement Hagree. destruct s as [x | sym | c]; simpl; try reflexivity.
  destruct (Pos.eqb_spec x v) as [-> | Hne].
  - destruct element as [y | sym | c]; simpl in Helement |- *;
      [ discriminate | symmetry; exact Helement | symmetry; exact Helement ].
  - simpl. apply Hagree, Hne.
Qed.

Lemma eval_simples_element_for (v : variable) (element : simple) (w : value)
    (vars vars' : env) (args : list simple) :
  eval_simple [] element = Some w ->
  (∀ x : variable, x <> v -> lookup x vars' = lookup x vars) ->
  eval_simples ((v, w) :: vars') args
  = eval_simples vars (map (element_for v element) args).
Proof.
  intros Helement Hagree.
  induction args as [ | s args IH]; simpl; [ reflexivity | ].
  rewrite (eval_simple_element_for v element w vars vars' s Helement Hagree),
    IH.
  reflexivity.
Qed.

(* So the merged application steps as the arm does. *)
Lemma merged_steps_as_loaded_arm (v : variable) (element : simple)
    (w : value) (merged : apply_cont) (vars vars' : env) (conts : cont_env)
    (statics : static_env) (hp : heap) (rd : read_record) :
  eval_simple [] element = Some w ->
  (∀ x : variable, x <> v -> lookup x vars' = lookup x vars) ->
  step (State (Apply_cont merged) ((v, w) :: vars') conts statics hp rd)
  = step
      (State
         (Apply_cont
            (Apply_cont_expr (ac_continuation merged)
               (map (element_for v element) (ac_args merged))
               (ac_trap_action merged)))
         vars conts statics hp rd).
Proof.
  intros Helement Hagree. rewrite !step_apply_cont. simpl.
  rewrite (eval_simples_element_for v element w vars vars' _ Helement Hagree).
  reflexivity.
Qed.

(* The arm that a [Switch] goes to is one that the test has looked at. *)
Lemma arm_is_loaded_found (v : variable) (merged : apply_cont)
    (first offset : Z) (elements : list simple) (immediates : bool)
    (arms : list (Z * apply_cont)) (i : Z) (ac : apply_cont) :
  forallb (arm_is_loaded v merged first offset elements immediates) arms
  = true ->
  find_arm i arms = Some ac ->
  ∃ element : simple,
    table_element first offset elements i = Some element
    ∧ element_taken immediates element = true
    ∧ ac
      = Apply_cont_expr (ac_continuation merged)
          (map (element_for v element) (ac_args merged))
          (ac_trap_action merged).
Proof.
  intros Hall Hfind. apply find_arm_in in Hfind.
  rewrite forallb_forall in Hall. specialize (Hall (i, ac) Hfind).
  unfold arm_is_loaded in Hall. simpl in Hall.
  destruct (table_element first offset elements i) as [element | ];
    [ | discriminate ].
  apply andb_prop in Hall as [Htaken Hsame].
  exists element. split; [ reflexivity | split; [ exact Htaken | ] ].
  destruct (apply_cont_eq_dec ac _) as [Heq | ]; [ exact Heq | discriminate ].
Qed.

(* ======================================================================== *)
(* The steps of the new [Let]s                                              *)
(* ======================================================================== *)

(* The [Let] of the offset, where there is one, binds the variable to
   the index. *)
Lemma bind_table_offset_equivalent (v : variable)
    (first offset i index : Z) (body : expr) (vars : env) (conts : cont_env)
    (statics : static_env) (hp : heap) (rd : read_record) :
  lookup v vars = Some (Value_const (Tagged_immediate i)) ->
  table_index first offset i = Some index ->
  ∃ vars' : env,
    lookup v vars' = Some (Value_const (Tagged_immediate index))
    ∧ (∀ x : variable, x <> v -> lookup x vars' = lookup x vars)
    ∧ in_range (int_width Standard_tagged_immediate) index = true
    ∧ equivalent
        (State (bind_table_offset v first offset body) vars conts statics hp rd)
        (State body vars' conts statics hp rd).
Proof.
  intros Hv Hindex. unfold table_index in Hindex. cbv zeta in Hindex.
  destruct (in_range (int_width Standard_naked_immediate) i) eqn:Hrange;
    [ | discriminate ].
  unfold bind_table_offset.
  destruct (Z.eqb_spec first 0) as [-> | Hfirst].
  - injection Hindex as <-. exists vars.
    split; [ exact Hv | split; [ reflexivity | split ] ];
      [ exact Hrange | apply equivalent_refl ].
  - destruct (in_range (int_width Standard_tagged_immediate) offset)
      eqn:Hoffset; [ | discriminate ].
    injection Hindex as <-.
    exists
      ((v,
        Value_const
          (Tagged_immediate
             (wrap (int_width Standard_tagged_immediate) (i + offset))))
       :: vars).
    split; [ simpl; rewrite Pos.eqb_refl; reflexivity | split; [ | split ] ].
    + intros x Hne. apply lookup_other_variable, Hne.
    + exact
        (wrap_in_range _ _ (int_width_positive Standard_tagged_immediate)).
    + apply step_equivalent.
      rewrite
        (step_let_no_location _ _ _ _ _ _ _ _ _ rd_
           [Value_const (Tagged_immediate i);
            Value_const (Tagged_immediate offset)]);
        [ | intros q; discriminate
          | cbn [eval_simples eval_simple]; rewrite Hv; reflexivity
          | intros l [Hl | [Hl | [ ]]]; discriminate Hl ].
      cbn [eval_prim eval_binary_primitive int_of_value].
      change (int_width Standard_naked_immediate)
        with (int_width Standard_tagged_immediate) in Hrange.
      rewrite Hrange, Hoffset. reflexivity.
Qed.

(* A value that the load kind takes is of kind [Value]. *)
Lemma array_element_ok_value (kind : array_kind) (w : value) :
  array_element_ok kind w = true -> kind_eqb (kind_of_value w) Value = true.
Proof.
  destruct kind; cbn [array_element_ok]; intros Hok; try exact Hok.
  unfold is_tagged_immediate in Hok.
  destruct w; simpl in Hok |- *; try discriminate; try reflexivity.
  match goal with
  | c : const |- _ =>
    destruct c; simpl in Hok |- *; try discriminate; reflexivity
  end.
Qed.

(* The load of an element of a static array (P1's law, in the form
   that is needed here). *)
Lemma array_load_of_static (statics : static_env) (s : symbol)
    (elements : list value) (kind : array_kind)
    (load_kind : array_load_kind) (mut : mutability) (index v : value)
    (i : Z) :
  lookup s (static_values statics) = Some (Static_immutable_array elements) ->
  int_of_value Standard_tagged_immediate index = Some i ->
  0 <= i < Z.of_nat (length elements) ->
  nth_error elements (Z.to_nat i) = Some v ->
  array_element_ok (array_kind_of_load_kind load_kind) v = true ->
  eval_prim statics (Binary (Array_load kind load_kind mut))
    [Value_symbol s; index]
  = Some v.
Proof.
  intros Hs Hi [Hlow Hhigh] Hnth Hok.
  cbn [eval_prim eval_binary_primitive array_of]. rewrite Hs.
  cbn beta iota. unfold array_loaded, array_index. rewrite Hi.
  cbn beta iota.
  apply Z.leb_le in Hlow. apply Z.ltb_lt in Hhigh. rewrite Hlow, Hhigh.
  cbn [andb]; cbn beta iota. rewrite Hnth. cbn beta iota.
  rewrite Hok. reflexivity.
Qed.

(* The values of the elements of a table, one for one. *)
Lemma eval_simples_nth (elements : list simple) :
  ∀ (vs : list value) (n : nat) (element : simple),
    eval_simples [] elements = Some vs ->
    nth_error elements n = Some element ->
    ∃ w : value, eval_simple [] element = Some w ∧ nth_error vs n = Some w.
Proof.
  induction elements as [ | element0 elements IH];
    intros vs n element Heval Hnth; [ destruct n; discriminate | ].
  simpl in Heval.
  destruct (eval_simple [] element0) as [w0 | ] eqn:Hw0; [ | discriminate ].
  destruct (eval_simples [] elements) as [ws | ] eqn:Hws; [ | discriminate ].
  injection Heval as <-.
  destruct n as [ | n]; simpl in Hnth |- *.
  - injection Hnth as <-. exists w0. split; [ exact Hw0 | reflexivity ].
  - exact (IH ws n element eq_refl Hnth).
Qed.

Lemma eval_simples_length (vars : env) (elements : list simple) :
  ∀ vs : list value,
    eval_simples vars elements = Some vs -> length vs = length elements.
Proof.
  induction elements as [ | element elements IH]; intros vs Heval;
    simpl in Heval.
  - injection Heval as <-. reflexivity.
  - destruct (eval_simple vars element); [ | discriminate ].
    destruct (eval_simples vars elements) as [ws | ]; [ | discriminate ].
    injection Heval as <-. simpl. f_equal. exact (IH ws eq_refl).
Qed.

(* An element that the test has passed is one that the load kind of the
   table takes. *)
Lemma element_taken_ok (elements : list simple) (element : simple)
    (w : value) :
  element ∈ elements ->
  eval_simple [] element = Some w ->
  element_taken (all_immediates elements) element = true ->
  array_element_ok
    (array_kind_of_load_kind (table_load_kind (all_immediates elements))) w
  = true.
Proof.
  intros Hin Hw Htaken.
  destruct (all_immediates elements) eqn:Himmediates;
    cbn [table_load_kind array_kind_of_load_kind].
  - unfold all_immediates in Himmediates.
    rewrite forallb_forall in Himmediates.
    specialize (Himmediates element Hin). rewrite Hw in Himmediates.
    exact Himmediates.
  - unfold element_taken in Htaken. cbn [orb] in Htaken.
    rewrite Hw in Htaken. exact Htaken.
Qed.

(* ======================================================================== *)
(* The refinement of a [Switch]                                             *)
(* ======================================================================== *)

Lemma merge_by_table_refines (v : variable) (merged : apply_cont)
    (first offset : Z) (arms : list (Z * apply_cont)) (s : symbol)
    (elements : list simple) (vs : list value) (vars : env) (conts : cont_env)
    (statics : static_env) (hp : heap) (rd : read_record) :
  forallb
    (arm_is_loaded v merged first offset elements (all_immediates elements))
    arms
  = true ->
  eval_simples [] elements = Some vs ->
  lookup s (static_values statics) = Some (Static_immutable_array vs) ->
  refines (State (Switch (Var v) arms) vars conts statics hp rd)
    (State
       (bind_table_load v s first offset (all_immediates elements)
          (Apply_cont merged))
       vars conts statics hp rd).
Proof.
  intros Hall Hvs Hs.
  destruct (step_switch_cases (Var v) arms vars conts statics hp rd)
    as [Hwrong | (w & i & ac & Hv & Hi & Hfind & Hstep)];
    [ apply step_undefined_refines, Hwrong | ].
  apply naked_immediate_of_value_some in Hi. subst w.
  destruct (arm_is_loaded_found _ _ _ _ _ _ _ _ _ Hall Hfind)
    as (element & Helement & Htaken & ->).
  unfold table_element in Helement.
  destruct (table_index first offset i) as [index | ] eqn:Hindex;
    [ | discriminate ].
  cbn beta iota in Helement.
  destruct ((0 <=? index) && (index <? Z.of_nat (length elements)))
    eqn:Hbounds; [ | discriminate ].
  apply andb_prop in Hbounds as [Hlow Hhigh].
  apply Z.leb_le in Hlow. apply Z.ltb_lt in Hhigh.
  destruct (eval_simples_nth elements vs _ element Hvs Helement)
    as (w & Hw & Hnth).
  apply (element_taken_ok elements element w
           (nth_error_In _ _ Helement) Hw) in Htaken.
  assert (Hrange : in_range (int_width Standard_naked_immediate) i = true).
  { unfold table_index in Hindex. cbv zeta in Hindex.
    destruct (in_range (int_width Standard_naked_immediate) i);
      [ reflexivity | discriminate ]. }
  apply equivalent_refines.
  eapply equivalent_trans; [ apply step_equivalent, Hstep | ].
  apply equivalent_sym. unfold bind_table_load.
  eapply equivalent_trans;
    [ apply step_equivalent, (step_bind_tagged v i);
      [ exact Hv | exact Hrange ]
    | ].
  destruct
    (bind_table_offset_equivalent v first offset i index
       (Let v Value
          (Prim
             (Binary
                (Array_load (table_array_kind (all_immediates elements))
                   (table_load_kind (all_immediates elements)) Immutable))
             [Symbol s; Var v])
          (Apply_cont merged))
       ((v, Value_const (Tagged_immediate i)) :: vars) conts statics hp rd)
    as (vars' & Hv' & Hagree & Hindex_range & Hequiv);
    [ simpl; rewrite Pos.eqb_refl; reflexivity | exact Hindex | ].
  eapply equivalent_trans; [ exact Hequiv | ].
  eapply equivalent_trans.
  - apply step_equivalent.
    rewrite
      (step_let_no_location _ _ _ _ _ _ _ _ _ rd_
         [Value_symbol s; Value_const (Tagged_immediate index)]);
      [ | intros q; discriminate
        | cbn [eval_simples eval_simple]; rewrite Hv'; reflexivity
        | intros l [Hl | [Hl | [ ]]]; discriminate Hl ].
    rewrite
      (array_load_of_static statics s vs _ _ Immutable
         (Value_const (Tagged_immediate index)) w index Hs);
      [ rewrite (array_element_ok_value _ w Htaken); reflexivity
      | cbn [int_of_value]; rewrite Hindex_range; reflexivity
      | rewrite (eval_simples_length [] elements vs Hvs); lia
      | exact Hnth | exact Htaken ].
  - apply step_eq_equivalent, merged_steps_as_loaded_arm; [ exact Hw | ].
    intros x Hne. rewrite (Hagree x Hne). apply lookup_other_variable, Hne.
Qed.

Lemma merge_switch_by_table_refines (tables : known_tables)
    (scrutinee : simple) (arms : list (Z * apply_cont)) (merged : expr)
    (vars : env) (conts : cont_env) (statics : static_env) (hp : heap)
      (rd : read_record) :
  tables_bound tables statics ->
  merge_switch_by_table tables scrutinee arms = Some merged ->
  refines (State (Switch scrutinee arms) vars conts statics hp rd)
    (State merged vars conts statics hp rd).
Proof.
  intros Hbound. unfold merge_switch_by_table.
  destruct scrutinee as [v | sym | c]; try discriminate.
  destruct arms as [ | [first ac] [ | arm1 [ | arm2 others]]];
    try discriminate.
  cbv zeta.
  destruct (var_in_simples v _); [ | discriminate ].
  destruct (find _ tables) as [[s elements] | ] eqn:Hfind; [ | discriminate ].
  intros [= <-].
  apply find_some in Hfind as [Hin Hhas].
  unfold table_has_arms in Hhas. apply andb_prop in Hhas as [_ Hall].
  cbv zeta in Hall. cbn [snd] in Hall.
  destruct (Hbound s elements Hin) as (vs & Hvs & Hs).
  exact (merge_by_table_refines v _ first _ _ s elements vs vars conts
    statics hp rd Hall Hvs Hs).
Qed.

(* With no table in scope no table has the arms. *)
Lemma merge_switch_by_table_nil (scrutinee : simple)
    (arms : list (Z * apply_cont)) :
  merge_switch_by_table [] scrutinee arms = None.
Proof.
  unfold merge_switch_by_table.
  destruct scrutinee as [v | sym | c]; try reflexivity.
  destruct arms as [ | [first ac] [ | arm1 [ | arm2 others]]];
    try reflexivity.
  cbv zeta. destruct (var_in_simples v _); reflexivity.
Qed.

(* ======================================================================== *)
(* The tables are bound                                                     *)
(* ======================================================================== *)

Lemma no_tables_bound (statics : static_env) : tables_bound [] statics.
Proof. intros s elements []. Qed.

Lemma tables_bound_extend (tables : known_tables)
    (statics statics' : static_env) :
  tables_bound tables statics ->
  statics_extend statics statics' ->
  tables_bound tables statics'.
Proof.
  intros Hbound [_ Hvalues] s elements Hin.
  destruct (Hbound s elements Hin) as (vs & Hvs & Hs).
  exists vs. split; [ exact Hvs | exact (Hvalues _ _ Hs) ].
Qed.

(* Elements with no variable have their values under any variables. *)
Lemma eval_closed_elements (vars : env) (elements : list simple) :
  forallb element_closed elements = true ->
  eval_simples vars elements = eval_simples [] elements.
Proof.
  induction elements as [ | element elements IH]; simpl; [ reflexivity | ].
  intros Hclosed. apply andb_prop in Hclosed as [Helement Hclosed].
  rewrite (IH Hclosed).
  destruct element as [x | sym | c]; try reflexivity. discriminate Helement.
Qed.

(* A group of static bindings that has a value binds its tables. *)
Lemma group_tables_bound (vars : env) (bindings : list static_binding) :
  ∀ (statics extended : static_env) (s : symbol) (elements : list simple),
    eval_static_bindings vars bindings statics = Some extended ->
    (s, elements) ∈ flat_map table_of_binding bindings ->
    ∃ vs : list value,
      eval_simples [] elements = Some vs
      ∧ lookup s (static_values extended) = Some (Static_immutable_array vs).
Proof.
  induction bindings as [ | binding bindings IH];
    intros statics extended s elements Heval Hin; [ destruct Hin | ].
  cbn [flat_map] in Hin. apply in_app_or in Hin.
  destruct binding as [id c | id | s0 const];
    cbn [eval_static_bindings] in Heval.
  - destruct (lookup id (static_code statics)); [ discriminate | ].
    destruct Hin as [[] | Hin]. exact (IH _ _ s elements Heval Hin).
  - destruct Hin as [[] | Hin]. exact (IH _ _ s elements Heval Hin).
  - destruct (lookup s0 (static_values statics)); [ discriminate | ].
    destruct (eval_static_const vars const) as [allocated | ] eqn:Hconst;
      [ | discriminate ].
    cbn beta iota in Heval.
    destruct Hin as [Hin | Hin]; [ | exact (IH _ _ s elements Heval Hin) ].
    destruct const as [ | | | | | elements0]; try destruct Hin.
    cbn [table_of_binding] in Hin.
    destruct (forallb element_closed elements0) eqn:Hclosed;
      [ | destruct Hin ].
    destruct Hin as [[= <- <-] | []].
    cbn [eval_static_const] in Hconst.
    rewrite (eval_closed_elements vars elements0 Hclosed) in Hconst.
    destruct (eval_simples [] elements0) as [vs | ]; [ | discriminate ].
    injection Hconst as <-.
    exists vs. split; [ reflexivity | ].
    destruct (eval_static_bindings_extend vars bindings _ extended Heval)
      as [_ Hvalues].
    apply Hvalues. simpl. rewrite Pos.eqb_refl. reflexivity.
Qed.

Lemma tables_under_statics_bound (tables : known_tables)
    (bindings : list static_binding) (vars : env)
    (statics extended : static_env) :
  tables_bound tables statics ->
  eval_static_bindings vars bindings statics = Some extended ->
  tables_bound (tables_under_statics tables bindings) extended.
Proof.
  intros Hbound Heval s elements Hin.
  unfold tables_under_statics in Hin. apply in_app_or in Hin.
  destruct Hin as [Hin | Hin].
  - exact (group_tables_bound vars bindings statics extended s elements Heval
      Hin).
  - exact (tables_bound_extend tables statics extended Hbound
      (eval_static_bindings_extend vars bindings statics extended Heval)
      s elements Hin).
Qed.

(* ======================================================================== *)
(* A way for each argument                                                  *)
(* ======================================================================== *)

(* An element that has a value with no variable has it under any. *)
Lemma eval_closed_element (vars : env) (element : simple) (w : value) :
  eval_simple [] element = Some w -> eval_simple vars element = Some w.
Proof. destruct element as [x | sym | c]; [ discriminate | | ]; exact id. Qed.

(* The tables of the ways are bound, and each is kept with whether
   its elements are all immediates. *)
Definition ways_bound (ways : list table_way) (statics : static_env) : Prop :=
  ∀ (s : symbol) (elements : list simple) (immediates : bool),
    From_table s elements immediates ∈ ways ->
    immediates = all_immediates elements
    ∧ ∃ vs : list value,
        eval_simples [] elements = Some vs
        ∧ lookup s (static_values statics) = Some (Static_immutable_array vs).

Lemma table_ways_of_arguments_bound (tables : known_tables)
    (first offset : Z) (statics : static_env) :
  tables_bound tables statics ->
  ∀ (count : nat) (rows : list (Z * list simple)) (ways : list table_way),
    table_ways_of_arguments tables first offset count rows = Some ways ->
    ways_bound ways statics.
Proof.
  intros Hbound.
  induction count as [ | count IH]; intros rows ways Hways;
    cbn [table_ways_of_arguments] in Hways.
  - injection Hways as <-. intros s elements immediates [].
  - destruct (column_of_arms rows) as [column | ]; [ | discriminate ].
    cbn beta iota in Hways.
    destruct (table_way_of_column tables first offset column) as [way | ]
      eqn:Hway; [ | discriminate ].
    cbn beta iota in Hways.
    destruct
      (table_ways_of_arguments tables first offset count (rest_of_arms rows))
      as [ways0 | ] eqn:Hways0; [ | discriminate ].
    injection Hways as <-.
    intros s elements immediates [Hin | Hin];
      [ | exact (IH _ _ Hways0 s elements immediates Hin) ].
    subst way. unfold table_way_of_column in Hway.
    destruct (way_of_column column); [ discriminate | ].
    destruct (find _ tables) as [[s0 elements0] | ] eqn:Hfind;
      [ | discriminate ].
    injection Hway as -> -> <-.
    apply find_some in Hfind as [Hin _].
    split; [ reflexivity | exact (Hbound s elements Hin) ].
Qed.

(* The [Let]s of a load bind the variable to the element of the table
   for the arm, and leave the other variables as they are. *)
Lemma bind_table_load_to_equivalent (v x : variable) (s : symbol)
    (first offset i : Z) (elements : list simple) (vs : list value)
    (element : simple) (body : expr) (vars : env) (conts : cont_env)
    (statics : static_env) (hp : heap) (rd : read_record) :
  lookup v vars = Some (Value_const (Naked_immediate i)) ->
  table_element first offset elements i = Some element ->
  element_taken (all_immediates elements) element = true ->
  eval_simples [] elements = Some vs ->
  lookup s (static_values statics) = Some (Static_immutable_array vs) ->
  ∃ (w : value) (vars' : env),
    eval_simple [] element = Some w
    ∧ (∀ y : variable, y <> x -> lookup y vars' = lookup y vars)
    ∧ lookup x vars' = Some w
    ∧ equivalent
        (State
           (bind_table_load_to v x s first offset (all_immediates elements)
              body)
           vars conts statics hp rd)
        (State body vars' conts statics hp rd).
Proof.
  intros Hv Helement Htaken Hvs Hs.
  pose proof Helement as Hnth_element. unfold table_element in Hnth_element.
  destruct (table_index first offset i) as [index | ] eqn:Hindex;
    [ | discriminate ].
  cbn beta iota in Hnth_element.
  destruct ((0 <=? index) && (index <? Z.of_nat (length elements)))
    eqn:Hbounds; [ | discriminate ].
  apply andb_prop in Hbounds as [Hlow Hhigh].
  apply Z.leb_le in Hlow. apply Z.ltb_lt in Hhigh.
  destruct (eval_simples_nth elements vs _ element Hvs Hnth_element)
    as (w & Hw & Hnth).
  pose proof
    (element_taken_ok elements element w (nth_error_In _ _ Hnth_element) Hw
       Htaken)
    as Hok.
  assert (Hrange : in_range (int_width Standard_naked_immediate) i = true).
  { unfold table_index in Hindex. cbv zeta in Hindex.
    destruct (in_range (int_width Standard_naked_immediate) i);
      [ reflexivity | discriminate ]. }
  unfold bind_table_load_to.
  destruct
    (bind_table_offset_equivalent x first offset i index
       (Let x Value
          (Prim
             (Binary
                (Array_load (table_array_kind (all_immediates elements))
                   (table_load_kind (all_immediates elements)) Immutable))
             [Symbol s; Var x])
          body)
       ((x, Value_const (Tagged_immediate i)) :: vars) conts statics hp rd)
    as (vars1 & Hx1 & Hagree1 & Hindex_range & Hequiv1);
    [ simpl; rewrite Pos.eqb_refl; reflexivity | exact Hindex | ].
  exists w, ((x, w) :: vars1).
  split; [ exact Hw | split; [ | split ] ].
  - intros y Hne.
    rewrite (lookup_other_variable y x _ _ Hne), (Hagree1 y Hne).
    apply lookup_other_variable, Hne.
  - simpl. rewrite Pos.eqb_refl. reflexivity.
  - eapply equivalent_trans;
      [ apply step_equivalent, (step_bind_tagged_to v x i);
        [ exact Hv | exact Hrange ]
      | ].
    eapply equivalent_trans; [ exact Hequiv1 | ].
    apply step_equivalent.
    rewrite
      (step_let_no_location _ _ _ _ _ _ _ _ _ rd_
         [Value_symbol s; Value_const (Tagged_immediate index)]);
      [ | intros q; discriminate
        | cbn [eval_simples eval_simple]; rewrite Hx1; reflexivity
        | intros l [Hl | [Hl | [ ]]]; discriminate Hl ].
    rewrite
      (array_load_of_static statics s vs _ _ Immutable
         (Value_const (Tagged_immediate index)) w index Hs);
      [ rewrite (array_element_ok_value _ w Hok); reflexivity
      | cbn [int_of_value]; rewrite Hindex_range; reflexivity
      | rewrite (eval_simples_length [] elements vs Hvs); lia
      | exact Hnth | exact Hok ].
Qed.

(* The arguments that are the same in every arm mention none of the
   variables of a list: the test of [table_names_usable]. *)
Definition table_same_avoid (avoid : list variable) (ways : list table_way)
    : bool :=
  forallb
    (fun way =>
       match way with
       | Without_table (Same_in_every_arm s) =>
         negb (existsb (fun x => var_in_simple x s) avoid)
       | Without_table (Computed _) | From_table _ _ _ => true
       end)
    ways.

(* So the arguments of an arm have the same values in two
   environments that differ on those variables only. *)
Lemma arguments_by_table_ways_eval (first offset i : Z)
    (avoid : list variable) (statics : static_env) (vars vars' : env)
    (ways : list table_way) :
  ways_bound ways statics ->
  ∀ args : list simple,
    arguments_by_table_ways first offset i ways = Some args ->
    table_same_avoid avoid ways = true ->
    (∀ y : variable, among y avoid = false -> lookup y vars' = lookup y vars) ->
    eval_simples vars' args = eval_simples vars args.
Proof.
  induction ways as [ | way ways IH]; intros Hbound args Hargs Hsame Hagree;
    cbn [arguments_by_table_ways] in Hargs.
  - injection Hargs as <-. reflexivity.
  - destruct (argument_by_table_way first offset i way) as [s | ] eqn:Hs;
      [ | discriminate ].
    cbn beta iota in Hargs.
    destruct (arguments_by_table_ways first offset i ways) as [args0 | ];
      [ | discriminate ].
    injection Hargs as <-.
    unfold table_same_avoid in Hsame. cbn [forallb] in Hsame.
    apply andb_prop in Hsame as [Hway Hsame].
    cbn [eval_simples].
    rewrite
      (IH
         (fun s0 elements immediates Hin =>
            Hbound s0 elements immediates (or_intror Hin))
         args0 eq_refl Hsame Hagree).
    replace (eval_simple vars' s) with (eval_simple vars s); [ reflexivity | ].
    destruct way as [[s0 | how] | s0 elements immediates];
      cbn [argument_by_table_way argument_in_arm] in Hs.
    + injection Hs as <-. symmetry.
      apply (eval_simple_avoiding avoid); [ | exact Hagree ].
      apply negb_true_iff, Hway.
    + destruct (constant_of_arm how i); [ | discriminate ].
      injection Hs as <-. reflexivity.
    + destruct (table_element first offset elements i) as [element | ]
        eqn:Helement; [ | discriminate ].
      cbn beta iota in Hs.
      destruct (element_taken _ element); [ | discriminate ].
      injection Hs as <-.
      destruct (Hbound s0 elements immediates (or_introl eq_refl))
        as (_ & vs & Hvs & _).
      unfold table_element in Helement.
      destruct (table_index first offset i); [ | discriminate ].
      cbn beta iota in Helement.
      destruct (_ && _); [ | discriminate ].
      destruct (eval_simples_nth elements vs _ element Hvs Helement)
        as (w & Hw & _).
      rewrite (eval_closed_element vars element w Hw),
        (eval_closed_element vars' element w Hw).
      reflexivity.
Qed.

(* The [Let]s for the arguments leave an environment in which the
   merged arguments have the values of the arguments of the arm. *)
Lemma bind_table_ways_equivalent (first offset : Z) (v : variable) (i : Z)
    (avoid : list variable) (body : expr) (conts : cont_env)
    (statics : static_env) (hp : heap) (rd : read_record)
      (names : list variable) :
  ∀ (ways : list table_way) (args : list simple) (vars : env),
    ways_bound ways statics ->
    lookup v vars = Some (Value_const (Naked_immediate i)) ->
    length names = length ways ->
    distinct names = true ->
    among v names = false ->
    (∀ x : variable, among x names = true -> among x avoid = true) ->
    table_same_avoid avoid ways = true ->
    arguments_by_table_ways first offset i ways = Some args ->
    ∃ vars' : env,
      (∀ y : variable, among y names = false -> lookup y vars' = lookup y vars)
      ∧ equivalent
          (State (bind_table_ways first offset v names ways body) vars conts
             statics hp rd)
          (State body vars' conts statics hp rd)
      ∧ eval_simples vars' (merged_by_table_ways v names ways)
        = eval_simples vars args.
Proof.
  induction names as [ | x names IH];
    intros [ | way ways] args vars Hbound Hv Hlength Hdistinct Hvnames Havoid
      Hsame Hargs;
    try discriminate.
  - injection Hargs as <-. exists vars.
    split; [ reflexivity | split; [ apply equivalent_refl | reflexivity ] ].
  - injection Hlength as Hlength.
    cbn [distinct] in Hdistinct.
    apply andb_prop in Hdistinct as [Hxnames Hdistinct].
    apply negb_true_iff in Hxnames.
    unfold among in Hvnames. cbn [existsb] in Hvnames.
    apply orb_false_elim in Hvnames as [Hvx Hvnames].
    apply Pos.eqb_neq in Hvx.
    assert (Hxavoid : among x avoid = true).
    { apply Havoid. unfold among. cbn [existsb]. rewrite Pos.eqb_refl.
      reflexivity. }
    assert
      (Havoid' :
        ∀ y : variable, among y names = true -> among y avoid = true).
    { intros y Hy. apply Havoid. unfold among in *. cbn [existsb].
      rewrite Hy. apply orb_true_r. }
    assert
      (Hoff :
        ∀ (vars1 vars2 : env),
          (∀ y : variable,
             among y names = false -> lookup y vars2 = lookup y vars1) ->
          ∀ y : variable,
            among y avoid = false -> lookup y vars2 = lookup y vars1).
    { intros vars1 vars2 Hagree y Hy. apply Hagree.
      destruct (among y names) eqn:Hin; [ | reflexivity ].
      rewrite (Havoid' y Hin) in Hy. discriminate. }
    assert (Hbound' : ways_bound ways statics).
    { intros s elements immediates Hin. apply Hbound. right. exact Hin. }
    pose proof Hsame as Hsame'.
    unfold table_same_avoid in Hsame'. cbn [forallb] in Hsame'.
    apply andb_prop in Hsame' as [Hway Hrest].
    cbn [arguments_by_table_ways] in Hargs.
    destruct (argument_by_table_way first offset i way) as [s0 | ] eqn:Hs0;
      [ | discriminate ].
    cbn beta iota in Hargs.
    destruct (arguments_by_table_ways first offset i ways) as [args0 | ]
      eqn:Hargs0; [ | discriminate ].
    injection Hargs as <-.
    destruct way as [[s | how] | s elements immediates];
      cbn [argument_by_table_way argument_in_arm] in Hs0.
    + injection Hs0 as <-.
      destruct
        (IH ways args0 vars Hbound' Hv Hlength Hdistinct Hvnames Havoid' Hrest
           Hargs0)
        as (vars' & Hagree & Hequiv & Heval).
      exists vars'. split.
      { intros y Hy. apply Hagree. unfold among in *. cbn [existsb] in Hy.
        apply orb_false_elim in Hy as [_ Hy]. exact Hy. }
      split; [ exact Hequiv | ].
      cbn [merged_by_table_ways merged_by_table_way eval_simples].
      rewrite Heval.
      rewrite (eval_simple_avoiding avoid s vars vars');
        [ reflexivity | apply negb_true_iff, Hway
        | exact (Hoff vars vars' Hagree) ].
    + destruct (constant_of_arm how i) as [c | ] eqn:Hc; [ | discriminate ].
      injection Hs0 as <-.
      destruct
        (bind_way_equivalent how v x i c
           (bind_table_ways first offset v names ways body) vars conts statics
           hp rd Hv Hc)
        as (vars1 & Hagree1 & Hx1 & Hequiv1).
      assert (Hv1 : lookup v vars1 = Some (Value_const (Naked_immediate i))).
      { rewrite (Hagree1 v Hvx). exact Hv. }
      destruct
        (IH ways args0 vars1 Hbound' Hv1 Hlength Hdistinct Hvnames Havoid'
           Hrest Hargs0)
        as (vars' & Hagree & Hequiv & Heval).
      exists vars'. split.
      { intros y Hy. unfold among in Hy. cbn [existsb] in Hy.
        apply orb_false_elim in Hy as [Hyx Hy].
        rewrite (Hagree y Hy). apply Hagree1, Pos.eqb_neq, Hyx. }
      split; [ exact (equivalent_trans _ _ _ Hequiv1 Hequiv) | ].
      cbn [bind_table_ways bind_table_way merged_by_table_ways eval_simples].
      rewrite Heval.
      rewrite
        (arguments_by_table_ways_eval first offset i avoid statics vars vars1
           ways Hbound' args0 Hargs0 Hrest).
      2: { intros y Hy. apply Hagree1. intros ->.
           rewrite Hxavoid in Hy. discriminate. }
      replace
        (eval_simple vars'
           (merged_by_table_way v x (Without_table (Computed how))))
        with (Some (Value_const c)); [ reflexivity | ].
      symmetry.
      destruct how as [ | | | kind offset' slope];
        cbn [merged_by_table_way eval_simple].
      * rewrite (Hagree x Hxnames). exact (Hx1 eq_refl).
      * rewrite (Hagree x Hxnames). exact (Hx1 eq_refl).
      * rewrite (Hagree v Hvnames), Hv1. cbn [constant_of_arm] in Hc.
        injection Hc as <-. reflexivity.
      * rewrite (Hagree x Hxnames). exact (Hx1 eq_refl).
    + destruct (Hbound s elements immediates (or_introl eq_refl))
        as (-> & vs & Hvs & Hs).
      destruct (table_element first offset elements i) as [element | ]
        eqn:Helement; [ | discriminate ].
      cbn beta iota in Hs0.
      destruct (element_taken (all_immediates elements) element) eqn:Htaken;
        [ | discriminate ].
      injection Hs0 as <-.
      destruct
        (bind_table_load_to_equivalent v x s first offset i elements vs
           element (bind_table_ways first offset v names ways body) vars conts
           statics hp rd Hv Helement Htaken Hvs Hs)
        as (w & vars1 & Hw & Hagree1 & Hx1 & Hequiv1).
      assert (Hv1 : lookup v vars1 = Some (Value_const (Naked_immediate i))).
      { rewrite (Hagree1 v Hvx). exact Hv. }
      destruct
        (IH ways args0 vars1 Hbound' Hv1 Hlength Hdistinct Hvnames Havoid'
           Hrest Hargs0)
        as (vars' & Hagree & Hequiv & Heval).
      exists vars'. split.
      { intros y Hy. unfold among in Hy. cbn [existsb] in Hy.
        apply orb_false_elim in Hy as [Hyx Hy].
        rewrite (Hagree y Hy). apply Hagree1, Pos.eqb_neq, Hyx. }
      split; [ exact (equivalent_trans _ _ _ Hequiv1 Hequiv) | ].
      cbn [bind_table_ways bind_table_way merged_by_table_ways
        merged_by_table_way eval_simples eval_simple].
      rewrite Heval.
      rewrite
        (arguments_by_table_ways_eval first offset i avoid statics vars vars1
           ways Hbound' args0 Hargs0 Hrest).
      2: { intros y Hy. apply Hagree1. intros ->.
           rewrite Hxavoid in Hy. discriminate. }
      rewrite (Hagree x Hxnames), Hx1,
        (eval_closed_element vars element w Hw).
      reflexivity.
Qed.

(* The arm that a [Switch] goes to is one that the test has looked at. *)
Lemma arm_has_table_ways_found (k : continuation) (first offset : Z)
    (ways : list table_way) (arms : list (Z * apply_cont)) (i : Z)
    (ac : apply_cont) :
  forallb (arm_has_table_ways k first offset ways) arms = true ->
  find_arm i arms = Some ac ->
  ∃ args : list simple,
    arguments_by_table_ways first offset i ways = Some args
    ∧ ac = Apply_cont_expr k args None.
Proof.
  intros Hall Hfind. apply find_arm_in in Hfind.
  rewrite forallb_forall in Hall. specialize (Hall (i, ac) Hfind).
  unfold arm_has_table_ways in Hall. simpl in Hall.
  destruct (arguments_by_table_ways first offset i ways) as [args | ];
    [ | discriminate ].
  exists args. split; [ reflexivity | ].
  destruct (apply_cont_eq_dec ac _) as [Heq | ]; [ exact Heq | discriminate ].
Qed.

Lemma merge_switch_by_table_ways_refines (known : known_continuations)
    (tables : known_tables) (scrutinee : simple)
    (arms : list (Z * apply_cont)) (merged : expr) (vars : env)
    (conts : cont_env) (statics : static_env) (hp : heap) (rd : read_record) :
  tables_bound tables statics ->
  merge_switch_by_table_ways known tables scrutinee arms = Some merged ->
  refines (State (Switch scrutinee arms) vars conts statics hp rd)
    (State merged vars conts statics hp rd).
Proof.
  intros Hbound. unfold merge_switch_by_table_ways.
  destruct scrutinee as [v | sym | c]; try discriminate.
  destruct arms as [ | [first ac0] [ | arm1 [ | arm2 others]]];
    try discriminate.
  cbv zeta.
  destruct (lookup (ac_continuation ac0) known) as [names | ];
    [ | discriminate ].
  cbn beta iota.
  destruct (table_ways_of_arguments _ _ _ _ _) as [ways | ] eqn:Hways;
    [ | discriminate ].
  cbn beta iota.
  destruct (existsb is_from_table ways && _ && _) eqn:Hok; [ | discriminate ].
  intros [= <-].
  apply andb_prop in Hok as [Hok Hall].
  apply andb_prop in Hok as [_ Husable].
  unfold table_names_usable in Husable.
  apply andb_prop in Husable as [Husable Hsame].
  apply andb_prop in Husable as [Husable Hvnames].
  apply andb_prop in Husable as [Hlength Hdistinct].
  apply Nat.eqb_eq in Hlength. apply negb_true_iff in Hvnames.
  pose proof
    (table_ways_of_arguments_bound tables first (table_offset first) statics
       Hbound _ _ ways Hways)
    as Hways_bound.
  destruct
    (step_switch_cases (Var v) ((first, ac0) :: arm1 :: arm2 :: others) vars
       conts statics hp rd)
    as [Hwrong | (w & i & ac & Hv & Hi & Hfind & Hstep)];
    [ apply step_undefined_refines, Hwrong | ].
  apply naked_immediate_of_value_some in Hi. subst w.
  destruct (arm_has_table_ways_found _ _ _ _ _ _ _ Hall Hfind)
    as (args & Hargs & ->).
  destruct
    (bind_table_ways_equivalent first (table_offset first) v i names
       (Apply_cont
          (Apply_cont_expr (ac_continuation ac0)
             (merged_by_table_ways v names ways) None))
       conts statics hp rd names ways args vars Hways_bound Hv Hlength Hdistinct
       Hvnames (fun x Hx => Hx) Hsame Hargs)
    as (vars' & _ & Hequiv & Heval).
  apply equivalent_refines.
  eapply equivalent_trans; [ apply step_equivalent, Hstep | ].
  apply equivalent_sym.
  eapply equivalent_trans; [ exact Hequiv | ].
  apply step_eq_equivalent. rewrite !step_apply_cont.
  cbn [ac_args ac_continuation]. rewrite Heval. reflexivity.
Qed.

(* With no table in scope no way is a table. *)
Lemma table_ways_without_tables (first offset : Z) :
  ∀ (count : nat) (rows : list (Z * list simple)) (ways : list table_way),
    table_ways_of_arguments [] first offset count rows = Some ways ->
    existsb is_from_table ways = false.
Proof.
  induction count as [ | count IH]; intros rows ways Hways;
    cbn [table_ways_of_arguments] in Hways.
  - injection Hways as <-. reflexivity.
  - destruct (column_of_arms rows) as [column | ]; [ | discriminate ].
    cbn beta iota in Hways.
    unfold table_way_of_column in Hways.
    destruct (way_of_column column) as [way | ]; [ | discriminate ].
    cbn beta iota in Hways.
    destruct
      (table_ways_of_arguments [] first offset count (rest_of_arms rows))
      as [ways0 | ] eqn:Hways0; [ | discriminate ].
    injection Hways as <-.
    cbn [existsb is_from_table]. exact (IH _ _ Hways0).
Qed.

Lemma merge_switch_by_table_ways_nil (known : known_continuations)
    (scrutinee : simple) (arms : list (Z * apply_cont)) :
  merge_switch_by_table_ways known [] scrutinee arms = None.
Proof.
  unfold merge_switch_by_table_ways.
  destruct scrutinee as [v | sym | c]; try reflexivity.
  destruct arms as [ | [first ac] [ | arm1 [ | arm2 others]]];
    try reflexivity.
  cbv zeta.
  destruct (lookup (ac_continuation ac) known) as [names | ];
    [ | reflexivity ].
  cbn beta iota.
  destruct (table_ways_of_arguments _ _ _ _ _) as [ways | ] eqn:Hways;
    [ | reflexivity ].
  cbn beta iota.
  rewrite (table_ways_without_tables _ _ _ _ ways Hways). reflexivity.
Qed.

(* The test for no table at all changes nothing of the answer. *)
Lemma merge_switch_with_tables_unfold (known : known_continuations)
    (tables : known_tables) (scrutinee : simple)
    (arms : list (Z * apply_cont)) :
  merge_switch_with_tables known tables scrutinee arms
  = match merge_switch_by_table tables scrutinee arms with
    | Some merged =>
      match merge_switch scrutinee arms with
      | Switch _ _ => merged
      | Let _ _ _ _ | Let_static _ _ | Let_cont_non_recursive _ _ _
      | Let_cont_recursive _ _ | Apply _ | Apply_cont _ | Invalid =>
        Switch scrutinee arms
      end
    | None =>
      match merge_switch_by_table_ways known tables scrutinee arms with
      | Some merged => merged
      | None => Switch scrutinee arms
      end
    end.
Proof.
  unfold merge_switch_with_tables.
  destruct tables as [ | table tables]; [ | reflexivity ].
  rewrite merge_switch_by_table_nil, merge_switch_by_table_ways_nil.
  reflexivity.
Qed.

Lemma merge_switch_with_tables_refines (known : known_continuations)
    (tables : known_tables) (scrutinee : simple)
    (arms : list (Z * apply_cont)) (vars : env) (conts : cont_env)
    (statics : static_env) (hp : heap) (rd : read_record) :
  tables_bound tables statics ->
  refines (State (Switch scrutinee arms) vars conts statics hp rd)
    (State (merge_switch_with_tables known tables scrutinee arms) vars conts
       statics hp rd).
Proof.
  intros Hbound. rewrite merge_switch_with_tables_unfold.
  destruct (merge_switch_by_table tables scrutinee arms) as [merged | ]
    eqn:Hmerged.
  - destruct (merge_switch scrutinee arms); try apply refines_refl.
    exact (merge_switch_by_table_refines tables scrutinee arms merged vars
      conts statics hp rd Hbound Hmerged).
  - destruct (merge_switch_by_table_ways known tables scrutinee arms)
      as [merged | ] eqn:Hways; [ | apply refines_refl ].
    exact (merge_switch_by_table_ways_refines known tables scrutinee arms
      merged vars conts statics hp rd Hbound Hways).
Qed.

(* ======================================================================== *)
(* Down a term                                                              *)
(* ======================================================================== *)

(* The rewrite leaves what binds something static as it is. *)
Lemma merge_switch_by_table_binds_statics (tables : known_tables)
    (scrutinee : simple) (arms : list (Z * apply_cont)) (merged : expr) :
  merge_switch_by_table tables scrutinee arms = Some merged ->
  binds_statics merged = false.
Proof.
  unfold merge_switch_by_table.
  destruct scrutinee as [v | sym | c]; try discriminate.
  destruct arms as [ | [first ac] [ | arm1 [ | arm2 others]]];
    try discriminate.
  cbv zeta.
  destruct (var_in_simples v _); [ | discriminate ].
  destruct (find _ tables) as [[s elements] | ]; [ | discriminate ].
  intros [= <-].
  unfold bind_table_load, bind_tagged, bind_table_offset.
  destruct (first =? 0); reflexivity.
Qed.

Lemma bind_arithmetic_steps_binds_statics (standard : standard_int)
    (v : variable) (body : expr) (steps : list arithmetic_step) :
  binds_statics (bind_arithmetic_steps standard v steps body)
  = binds_statics body.
Proof.
  induction steps as [ | step steps IH]; [ reflexivity | ].
  cbn [bind_arithmetic_steps]. destruct step; exact IH.
Qed.

Lemma bind_table_way_binds_statics (first offset : Z) (v x : variable)
    (way : table_way) (body : expr) :
  binds_statics (bind_table_way first offset v x way body)
  = binds_statics body.
Proof.
  destruct way as [[s | how] | s elements immediates]; cbn [bind_table_way];
    [ reflexivity | | ].
  - destruct how as [ | | | kind offset' slope]; try reflexivity.
    cbn [bind_way]. cbv zeta.
    destruct kind; cbn [bind_tagged_to bind_same_to binds_statics];
      apply bind_arithmetic_steps_binds_statics.
  - unfold bind_table_load_to, bind_tagged_to, bind_table_offset.
    destruct (first =? 0); reflexivity.
Qed.

Lemma bind_table_ways_binds_statics (first offset : Z) (v : variable)
    (body : expr) (names : list variable) :
  ∀ ways : list table_way,
    binds_statics (bind_table_ways first offset v names ways body)
    = binds_statics body.
Proof.
  induction names as [ | x names IH]; intros [ | way ways]; try reflexivity.
  cbn [bind_table_ways]. rewrite bind_table_way_binds_statics. apply IH.
Qed.

Lemma merge_switch_by_table_ways_binds_statics (known : known_continuations)
    (tables : known_tables) (scrutinee : simple)
    (arms : list (Z * apply_cont)) (merged : expr) :
  merge_switch_by_table_ways known tables scrutinee arms = Some merged ->
  binds_statics merged = false.
Proof.
  unfold merge_switch_by_table_ways.
  destruct scrutinee as [v | sym | c]; try discriminate.
  destruct arms as [ | [first ac] [ | arm1 [ | arm2 others]]];
    try discriminate.
  cbv zeta.
  destruct (lookup (ac_continuation ac) known) as [names | ];
    [ | discriminate ].
  cbn beta iota.
  destruct (table_ways_of_arguments _ _ _ _ _) as [ways | ];
    [ | discriminate ].
  cbn beta iota.
  destruct (_ && _ && _); [ | discriminate ].
  intros [= <-]. rewrite bind_table_ways_binds_statics. reflexivity.
Qed.

Lemma merge_switch_with_tables_binds_statics (known : known_continuations)
    (tables : known_tables) (scrutinee : simple)
    (arms : list (Z * apply_cont)) :
  binds_statics (merge_switch_with_tables known tables scrutinee arms)
  = false.
Proof.
  rewrite merge_switch_with_tables_unfold.
  destruct (merge_switch_by_table tables scrutinee arms) as [merged | ]
    eqn:Hmerged.
  - destruct (merge_switch scrutinee arms); try reflexivity.
    exact (merge_switch_by_table_binds_statics tables scrutinee arms merged
      Hmerged).
  - destruct (merge_switch_by_table_ways known tables scrutinee arms)
      as [merged | ] eqn:Hways; [ | reflexivity ].
    exact (merge_switch_by_table_ways_binds_statics known tables scrutinee
      arms merged Hways).
Qed.

Fixpoint merge_tables_under_binds_statics (e : expr) :
  ∀ (known : known_continuations) (tables : known_tables),
    binds_statics (merge_tables_under known tables e) = binds_statics e.
Proof.
  destruct e
    as [ x kind d body | bindings body | k h body | handlers body
       | ap | ac | scrutinee arms | ];
    intros known tables;
    cbn [merge_tables_under merge_tables_in_handler binds_statics];
    try reflexivity.
  - apply merge_tables_under_binds_statics.
  - rewrite merge_tables_under_binds_statics. reflexivity.
  - destruct h as [params metadata handler].
    cbn [merge_tables_in_handler handler_binds_statics].
    rewrite !merge_tables_under_binds_statics. reflexivity.
  - cbv zeta. rewrite merge_tables_under_binds_statics. f_equal.
    generalize
      (map (fun entry => (fst entry, parameters_of_handler (snd entry)))
         handlers
       ++ known).
    intros known'.
    induction handlers as [ | [k [params metadata handler]] handlers IH];
      simpl; [ reflexivity | ].
    rewrite merge_tables_under_binds_statics, IH. reflexivity.
  - apply merge_switch_with_tables_binds_statics.
Qed.

Lemma merge_tables_under_handlers_bind_statics (known : known_continuations)
    (tables : known_tables)
    (handlers : list (continuation * continuation_handler)) :
  handlers_bind_statics
    (map (fun '(k, h) => (k, merge_tables_in_handler known tables h))
       handlers)
  = handlers_bind_statics handlers.
Proof.
  unfold handlers_bind_statics.
  induction handlers as [ | [k [params metadata handler]] handlers IH]; simpl;
    [ reflexivity | ].
  rewrite merge_tables_under_binds_statics, IH. reflexivity.
Qed.

Fixpoint merge_tables_under_refines_aux (e : expr) :
  ∀ (known : known_continuations) (tables : known_tables) (vars : env)
    (conts : cont_env) (statics : static_env) (hp : heap) (rd : read_record),
    tables_bound tables statics ->
    refines (State e vars conts statics hp rd)
      (State (merge_tables_under known tables e) vars conts statics hp rd).
Proof.
  destruct e
    as [ x kind d body | bindings body | k h body | handlers body
       | ap | ac | scrutinee arms | ];
    intros known tables vars conts statics hp rd Hbound;
    cbn [merge_tables_under]; try apply refines_refl.
  - apply let_refines_step_exec; [ reflexivity | intros v hp' rd' _ _ ].
    apply merge_tables_under_refines_aux, Hbound.
  - apply let_static_refines_step.
    destruct (eval_static_bindings vars bindings statics) as [extended | ]
      eqn:Hbindings; [ | exact I ].
    apply merge_tables_under_refines_aux.
    exact (tables_under_statics_bound tables bindings vars statics extended
      Hbound Hbindings).
  - destruct h as [params metadata handler].
    cbn [merge_tables_in_handler].
    apply (refines_trans _
      (State
         (Let_cont_non_recursive k
            (Handler params metadata
               (merge_tables_under known tables handler))
            body)
         vars conts statics hp rd)).
    + apply (let_cont_handler_refines_where (tables_bound tables)
        (tables_bound_extend tables)); [ exact Hbound | ].
      intros vs handler_vars conts' statics' hp' rd' Hbound' _ _.
      apply merge_tables_under_refines_aux, Hbound'.
    + apply let_cont_non_recursive_refines_step,
        merge_tables_under_refines_aux, Hbound.
  - cbv zeta.
    generalize
      (map (fun entry => (fst entry, parameters_of_handler (snd entry)))
         handlers
       ++ known).
    intros known'.
    apply (refines_trans _
      (State
         (Let_cont_recursive
            (map (fun '(k, h) => (k, merge_tables_in_handler known' tables h))
               handlers)
            body)
         vars conts statics hp rd)).
    + assert (Hhandlers :
        handlers_refine (tables_bound tables) vars handlers
          (map (fun '(k, h) => (k, merge_tables_in_handler known' tables h))
             handlers)).
      { induction handlers
          as [ | [k [params metadata handler]] handlers IH];
          simpl.
        - split; apply Forall2_nil.
        - destruct IH as [Hforwards Hbackwards].
          split; (apply Forall2_cons; [ | assumption ]);
            (split; [ reflexivity | ]); apply Handler_related;
            intros vs handler_vars conts' statics' hp' rd' Hbound' _ _;
            apply merge_tables_under_refines_aux, Hbound'. }
      destruct Hhandlers as [Hforwards Hbackwards].
      apply (let_cont_recursive_handlers_refine_where
        (tables_bound tables) (tables_bound_extend tables));
        try assumption.
      symmetry. apply merge_tables_under_handlers_bind_statics.
    + apply let_cont_recursive_refines_step; [ reflexivity | intros _ ].
      apply merge_tables_under_refines_aux, Hbound.
  - apply merge_switch_with_tables_refines, Hbound.
Qed.

Theorem merge_tables_under_refines : merge_tables_under_refines_statement.
Proof.
  intros known tables e vars conts statics hp rd Hbound.
  apply merge_tables_under_refines_aux, Hbound.
Qed.

Lemma merge_tables_knowing_binds_statics (e : expr) (tables : known_tables) :
  binds_statics (merge_tables_knowing tables e) = binds_statics e.
Proof. apply merge_tables_under_binds_statics. Qed.

Lemma merge_tables_knowing_refines_aux (e : expr) (tables : known_tables)
    (vars : env) (conts : cont_env) (statics : static_env) (hp : heap)
      (rd : read_record) :
  tables_bound tables statics ->
  refines (State e vars conts statics hp rd)
    (State (merge_tables_knowing tables e) vars conts statics hp rd).
Proof. apply merge_tables_under_refines_aux. Qed.

Theorem merge_tables_knowing_refines : merge_tables_knowing_refines_statement.
Proof.
  intros tables e vars conts statics hp rd Hbound.
  apply merge_tables_knowing_refines_aux, Hbound.
Qed.

(* ======================================================================== *)
(* In the bodies of code                                                    *)
(* ======================================================================== *)

(* The tables of a group are those of the group with its code
   rewritten. *)
Lemma tables_under_rewritten_code (g : expr -> expr) (tables : known_tables)
    (bindings : list static_binding) :
  tables_under_statics tables (map (rewrite_code_in_binding g) bindings)
  = tables_under_statics tables bindings.
Proof.
  unfold tables_under_statics. f_equal.
  induction bindings as [ | binding bindings IH]; [ reflexivity | ].
  cbn [map flat_map]. rewrite IH.
  destruct binding
    as [id [params my_closure ret exn metadata body] | id | s const];
    reflexivity.
Qed.

(* The code of a group refines its counterparts where the tables are
   bound. *)
Lemma merged_bindings_refine (tables : known_tables)
    (bindings : list static_binding) :
  bindings_refine (tables_bound tables) bindings
    (map (rewrite_code_in_binding (merge_tables_knowing tables)) bindings).
Proof.
  induction bindings
    as [ | [id [params my_closure ret exn metadata code_body] | id
           | sym const]
           bindings [Hforwards Hbackwards]];
    simpl.
  - split; apply Forall2_nil.
  - unfold rewrite_code_body.
    destruct (Bool.eqb (binds_statics code_body)
                (binds_statics (merge_tables_knowing tables code_body)))
      eqn:Hbinds;
      [ apply eqb_prop in Hbinds | ];
      split; (apply Forall2_cons; [ | assumption ]);
      try apply Binding_same;
      apply Binding_code, Codes_related;
      first
        [ exact Hbinds | exact (eq_sym Hbinds)
        | intros vars conts statics Hbound;
          apply merge_tables_knowing_refines_aux; assumption ].
  - split; (apply Forall2_cons; [ apply Binding_same | assumption ]).
  - split; (apply Forall2_cons; [ apply Binding_same | assumption ]).
Qed.

Fixpoint merge_tables_in_code_binds_statics (e : expr) :
  ∀ tables : known_tables,
    binds_statics (merge_tables_in_code tables e) = binds_statics e.
Proof.
  destruct e
    as [ x kind d body | bindings body | k h body | handlers body
       | ap | ac | scrutinee arms | ];
    intros tables;
    cbn [merge_tables_in_code merge_tables_in_code_in_handler binds_statics];
    cbv zeta; try reflexivity.
  - apply merge_tables_in_code_binds_statics.
  - rewrite merge_tables_in_code_binds_statics. f_equal.
    generalize
      (merge_tables_knowing (tables_under_statics tables bindings)).
    intros g.
    induction bindings
      as [ | [id [params my_closure ret exn metadata code_body] | id
             | sym const]
             bindings IH];
      simpl; congruence.
  - destruct h as [params metadata handler].
    cbn [merge_tables_in_code_in_handler handler_binds_statics].
    rewrite !merge_tables_in_code_binds_statics. reflexivity.
  - rewrite merge_tables_in_code_binds_statics. f_equal.
    induction handlers as [ | [k [params metadata handler]] handlers IH];
      simpl; [ reflexivity | ].
    rewrite merge_tables_in_code_binds_statics, IH. reflexivity.
Qed.

Lemma merge_tables_in_code_handlers_bind_statics (tables : known_tables)
    (handlers : list (continuation * continuation_handler)) :
  handlers_bind_statics
    (map (fun '(k, h) => (k, merge_tables_in_code_in_handler tables h))
       handlers)
  = handlers_bind_statics handlers.
Proof.
  unfold handlers_bind_statics.
  induction handlers as [ | [k [params metadata handler]] handlers IH]; simpl;
    [ reflexivity | ].
  rewrite merge_tables_in_code_binds_statics, IH. reflexivity.
Qed.

(* Down the term, with the static environment binding the tables: at
   once in a body, and from then on in a handler and in code. *)
Fixpoint merge_tables_in_code_refines_aux (e : expr) :
  ∀ (tables : known_tables) (vars : env) (conts : cont_env)
    (statics : static_env) (hp : heap) (rd : read_record),
    tables_bound tables statics ->
    refines (State e vars conts statics hp rd)
      (State (merge_tables_in_code tables e) vars conts statics hp rd).
Proof.
  destruct e
    as [ x kind d body | bindings body | k h body | handlers body
       | ap | ac | scrutinee arms | ];
    intros tables vars conts statics hp rd Hbound;
    cbn [merge_tables_in_code]; cbv zeta; try apply refines_refl.
  - apply let_refines_step_exec; [ reflexivity | intros v hp' rd' _ _ ].
    apply merge_tables_in_code_refines_aux, Hbound.
  - (* the code of the group is rewritten with the tables of the group
       in scope: the body under the one static environment refines it
       under the other; then down the body *)
    set (tables' := tables_under_statics tables bindings).
    destruct (merged_bindings_refine tables' bindings)
      as [Hforwards Hbackwards].
    apply (fun H =>
      eval_static_bindings_related _ vars _ _ H statics statics
        (statics_related_refl _ statics)) in Hforwards, Hbackwards.
    apply let_static_refines_step.
    destruct (eval_static_bindings vars bindings statics) as [extended | ]
      eqn:Hbindings,
      (eval_static_bindings vars
         (map (rewrite_code_in_binding (merge_tables_knowing tables'))
            bindings)
         statics)
      as [extended' | ] eqn:Hbindings';
      try contradiction; [ | exact I ].
    assert (Hbound' : tables_bound tables' extended).
    { exact (tables_under_statics_bound tables bindings vars statics extended
        Hbound Hbindings). }
    assert (Hbound'' : tables_bound tables' extended').
    { unfold tables' at 1.
      rewrite <-
        (tables_under_rewritten_code (merge_tables_knowing tables') tables
           bindings).
      exact (tables_under_statics_bound tables _ vars statics extended'
        Hbound Hbindings'). }
    apply (refines_trans _ (State body vars conts extended' hp rd)).
    + apply (related_states_refines (tables_bound tables')
        (tables_bound_extend tables'));
        [ exact Hbound' | exact Hbound'' | apply cont_env_related_refl
        | apply cont_env_related_refl | exact Hforwards | exact Hbackwards ].
    + apply merge_tables_in_code_refines_aux, Hbound''.
  - destruct h as [params metadata handler].
    cbn [merge_tables_in_code_in_handler].
    apply (refines_trans _
      (State
         (Let_cont_non_recursive k
            (Handler params metadata (merge_tables_in_code tables handler))
            body)
         vars conts statics hp rd)).
    + apply (let_cont_handler_refines_where (tables_bound tables)
        (tables_bound_extend tables)); [ exact Hbound | ].
      intros vs handler_vars conts' statics' hp' rd' Hbound' _ _.
      apply merge_tables_in_code_refines_aux, Hbound'.
    + apply let_cont_non_recursive_refines_step,
        merge_tables_in_code_refines_aux, Hbound.
  - apply (refines_trans _
      (State
         (Let_cont_recursive
            (map
               (fun '(k, h) =>
                  (k, merge_tables_in_code_in_handler tables h))
               handlers)
            body)
         vars conts statics hp rd)).
    + assert (Hhandlers :
        handlers_refine (tables_bound tables) vars handlers
          (map
             (fun '(k, h) => (k, merge_tables_in_code_in_handler tables h))
             handlers)).
      { induction handlers
          as [ | [k [params metadata handler]] handlers IH];
          simpl.
        - split; apply Forall2_nil.
        - destruct IH as [Hforwards Hbackwards].
          split; (apply Forall2_cons; [ | assumption ]);
            (split; [ reflexivity | ]); apply Handler_related;
            intros vs handler_vars conts' statics' hp' rd' Hbound' _ _;
            apply merge_tables_in_code_refines_aux, Hbound'. }
      destruct Hhandlers as [Hforwards Hbackwards].
      apply (let_cont_recursive_handlers_refine_where
        (tables_bound tables) (tables_bound_extend tables));
        try assumption.
      symmetry. apply merge_tables_in_code_handlers_bind_statics.
    + apply let_cont_recursive_refines_step; [ reflexivity | intros _ ].
      apply merge_tables_in_code_refines_aux, Hbound.
Qed.

Theorem merge_tables_in_code_refines : merge_tables_in_code_refines_statement.
Proof.
  intros tables e vars conts statics hp rd Hbound.
  apply merge_tables_in_code_refines_aux, Hbound.
Qed.

Theorem merge_tables_refines : merge_tables_refines_statement.
Proof.
  intros e vars conts statics hp rd. unfold merge_tables.
  eapply refines_trans.
  - apply merge_tables_knowing_refines_aux, no_tables_bound.
  - apply merge_tables_in_code_refines_aux, no_tables_bound.
Qed.

(* ======================================================================== *)
(* Renaming                                                                 *)
(* ======================================================================== *)

(* The elements of the tables in scope have no variable. *)
Definition tables_closed (tables : known_tables) : Prop :=
  ∀ (s : symbol) (elements : list simple),
    (s, elements) ∈ tables -> forallb element_closed elements = true.

Lemma element_closed_renamed (r : renaming) (s : simple) :
  element_closed (rename_simple r s) = element_closed s.
Proof. destruct s; reflexivity. Qed.

Lemma closed_element_renamed (r : renaming) (s : simple) :
  element_closed s = true -> rename_simple r s = s.
Proof. destruct s; [ discriminate | reflexivity | reflexivity ]. Qed.

Lemma closed_elements_renamed (r : renaming) (elements : list simple) :
  forallb element_closed elements = true ->
  map (rename_simple r) elements = elements.
Proof.
  induction elements as [ | element elements IH]; simpl; [ reflexivity | ].
  intros Hclosed. apply andb_prop in Hclosed as [Helement Hclosed].
  rewrite (closed_element_renamed r element Helement), (IH Hclosed).
  reflexivity.
Qed.

Lemma table_of_binding_renamed (r : renaming) (binding : static_binding) :
  table_of_binding (rename_static_binding r binding)
  = table_of_binding binding.
Proof.
  destruct binding as [id c | id | s const]; try reflexivity.
  destruct const as [ | | | | | elements]; try reflexivity.
  cbn [rename_static_binding rename_static_const table_of_binding].
  rewrite (forallb_under_map _ _ element_closed)
    by apply element_closed_renamed.
  destruct (forallb element_closed elements) eqn:Hclosed; [ | reflexivity ].
  rewrite (closed_elements_renamed r elements Hclosed). reflexivity.
Qed.

Lemma tables_under_statics_renamed (r : renaming) (tables : known_tables)
    (bindings : list static_binding) :
  tables_under_statics tables (map (rename_static_binding r) bindings)
  = tables_under_statics tables bindings.
Proof.
  unfold tables_under_statics. f_equal.
  induction bindings as [ | binding bindings IH]; [ reflexivity | ].
  cbn [map flat_map]. rewrite table_of_binding_renamed, IH. reflexivity.
Qed.

Lemma no_tables_closed : tables_closed [].
Proof. intros s elements []. Qed.

Lemma tables_under_statics_closed (tables : known_tables)
    (bindings : list static_binding) :
  tables_closed tables ->
  tables_closed (tables_under_statics tables bindings).
Proof.
  intros Hclosed s elements Hin. unfold tables_under_statics in Hin.
  apply in_app_or in Hin as [Hin | Hin]; [ | exact (Hclosed s elements Hin) ].
  apply in_flat_map in Hin as (binding & _ & Hin).
  destruct binding as [id c | id | s0 const]; try destruct Hin.
  destruct const as [ | | | | | elements0]; try destruct Hin.
  cbn [table_of_binding] in Hin.
  destruct (forallb element_closed elements0) eqn:Hclosed0;
    [ | destruct Hin ].
  destruct Hin as [[= <- <-] | []]. exact Hclosed0.
Qed.

Lemma element_for_renamed (r : renaming) (v : variable) (element : simple)
    (s : simple) :
  element_closed element = true ->
  element_for (rename_variable r v) element (rename_simple r s)
  = rename_simple r (element_for v element s).
Proof.
  intros Hclosed. destruct s as [x | sym | c]; simpl; try reflexivity.
  rewrite rename_variable_eqb.
  destruct (Pos.eqb x v); [ | reflexivity ].
  symmetry. exact (closed_element_renamed r element Hclosed).
Qed.

Lemma elements_for_renamed (r : renaming) (v : variable) (element : simple)
    (args : list simple) :
  element_closed element = true ->
  map (element_for (rename_variable r v) element)
    (map (rename_simple r) args)
  = map (rename_simple r) (map (element_for v element) args).
Proof.
  intros Hclosed. rewrite !map_map. apply map_ext. intros s.
  apply element_for_renamed, Hclosed.
Qed.

Lemma table_element_closed (first offset : Z) (elements : list simple)
    (i : Z) (element : simple) :
  forallb element_closed elements = true ->
  table_element first offset elements i = Some element ->
  element_closed element = true.
Proof.
  intros Hclosed Helement. unfold table_element in Helement.
  destruct (table_index first offset i) as [index | ]; [ | discriminate ].
  cbn beta iota in Helement.
  destruct (_ && _); [ | discriminate ].
  apply nth_error_In in Helement.
  rewrite forallb_forall in Hclosed. exact (Hclosed element Helement).
Qed.

Lemma arm_is_loaded_renamed (r : renaming) (v : variable)
    (merged : apply_cont) (first offset : Z) (elements : list simple)
    (immediates : bool) (arm : Z * apply_cont) :
  forallb element_closed elements = true ->
  arm_is_loaded (rename_variable r v) (rename_apply_cont r merged) first
    offset elements immediates (fst arm, rename_apply_cont r (snd arm))
  = arm_is_loaded v merged first offset elements immediates arm.
Proof.
  intros Hclosed. unfold arm_is_loaded. cbn [fst snd].
  destruct (table_element first offset elements (fst arm)) as [element | ]
    eqn:Helement; [ | reflexivity ].
  f_equal.
  set (target :=
    Apply_cont_expr (ac_continuation merged)
      (map (element_for v element) (ac_args merged))
      (ac_trap_action merged)).
  replace
    (Apply_cont_expr (ac_continuation (rename_apply_cont r merged))
       (map (element_for (rename_variable r v) element)
          (ac_args (rename_apply_cont r merged)))
       (ac_trap_action (rename_apply_cont r merged)))
    with (rename_apply_cont r target).
  - destruct (apply_cont_eq_dec (snd arm) target) as [-> | Hne],
      (apply_cont_eq_dec (rename_apply_cont r _) (rename_apply_cont r _))
        as [Heq | Hne'];
      try reflexivity; [ contradiction | ].
    elim Hne. exact (renamed_apply_cont_same r _ _ Heq).
  - unfold target, rename_apply_cont. cbn [ac_continuation ac_args
      ac_trap_action].
    rewrite elements_for_renamed; [ reflexivity | ].
    exact (table_element_closed first offset elements _ element Hclosed
      Helement).
Qed.

Lemma table_has_arms_renamed (r : renaming) (v : variable)
    (merged : apply_cont) (first offset : Z) (arms : list (Z * apply_cont))
    (table : symbol * list simple) :
  forallb element_closed (snd table) = true ->
  table_has_arms (rename_variable r v) (rename_apply_cont r merged) first
    offset (map (fun arm => (fst arm, rename_apply_cont r (snd arm))) arms)
    table
  = table_has_arms v merged first offset arms table.
Proof.
  intros Hclosed. unfold table_has_arms. rewrite length_map. f_equal.
  cbv zeta. apply forallb_under_map. intros arm.
  apply arm_is_loaded_renamed, Hclosed.
Qed.

Lemma merged_across_renamed (r : renaming) (v : variable)
    (arms : list (Z * apply_cont)) :
  ∀ args : list simple,
    merged_across (rename_variable r v) (map (rename_simple r) args)
      (map (fun arm => (fst arm, rename_apply_cont r (snd arm))) arms)
    = map (rename_simple r) (merged_across v args arms).
Proof.
  induction arms as [ | arm arms IH]; intros args; [ reflexivity | ].
  cbn [map merged_across snd].
  change (ac_args (rename_apply_cont r (snd arm)))
    with (map (rename_simple r) (ac_args (snd arm))).
  rewrite merged_arguments_renamed. apply IH.
Qed.

Lemma find_same_among {A : Type} (f g : A -> bool) (l : list A) :
  (∀ a : A, a ∈ l -> f a = g a) -> find f l = find g l.
Proof.
  induction l as [ | a l IH]; intros Hsame; [ reflexivity | ].
  cbn [find]. rewrite (Hsame a (or_introl eq_refl)).
  destruct (g a); [ reflexivity | ].
  apply IH. intros a' Hin. apply Hsame. right. exact Hin.
Qed.

Lemma bind_table_load_renamed (r : renaming) (v : variable) (s : symbol)
    (first offset : Z) (immediates : bool) (body : expr) :
  bind_table_load (rename_variable r v) s first offset immediates
    (rename_expr r body)
  = rename_expr r (bind_table_load v s first offset immediates body).
Proof.
  unfold bind_table_load, bind_tagged, bind_table_offset.
  destruct (first =? 0); reflexivity.
Qed.

Lemma merge_switch_by_table_renamed (r : renaming) (tables : known_tables)
    (scrutinee : simple) (arms : list (Z * apply_cont)) :
  tables_closed tables ->
  merge_switch_by_table tables (rename_simple r scrutinee)
    (map (fun arm => (fst arm, rename_apply_cont r (snd arm))) arms)
  = option_map (rename_expr r) (merge_switch_by_table tables scrutinee arms).
Proof.
  intros Hclosed.
  destruct scrutinee as [v | sym | c]; try reflexivity.
  destruct arms as [ | [first ac] [ | arm1 [ | arm2 others]]];
    try reflexivity.
  unfold merge_switch_by_table. cbn [map fst snd rename_simple]. cbv zeta.
  set (merged :=
    Apply_cont_expr (ac_continuation ac)
      (merged_across v (ac_args ac) (arm1 :: arm2 :: others)) None).
  replace
    (Apply_cont_expr (ac_continuation (rename_apply_cont r ac))
       (merged_across (rename_variable r v)
          (ac_args (rename_apply_cont r ac))
          ((fst arm1, rename_apply_cont r (snd arm1))
           :: (fst arm2, rename_apply_cont r (snd arm2))
           :: map (fun arm => (fst arm, rename_apply_cont r (snd arm)))
                others))
       None)
    with (rename_apply_cont r merged)
    by (unfold merged, rename_apply_cont;
        cbn [ac_continuation ac_args ac_trap_action option_map];
        rewrite <- (merged_across_renamed r v (arm1 :: arm2 :: others));
        reflexivity).
  change (ac_args (rename_apply_cont r merged))
    with (map (rename_simple r) (ac_args merged)).
  rewrite var_in_simples_renamed.
  destruct (var_in_simples v (ac_args merged)); [ | reflexivity ].
  change
    ((first, rename_apply_cont r ac)
     :: (fst arm1, rename_apply_cont r (snd arm1))
     :: (fst arm2, rename_apply_cont r (snd arm2))
     :: map (fun arm => (fst arm, rename_apply_cont r (snd arm))) others)
    with
      (map (fun arm => (fst arm, rename_apply_cont r (snd arm)))
         ((first, ac) :: arm1 :: arm2 :: others)).
  rewrite
    (find_same_among _
       (table_has_arms v merged first (table_offset first)
          ((first, ac) :: arm1 :: arm2 :: others))
       tables).
  - destruct (find _ tables) as [[s elements] | ]; [ | reflexivity ].
    cbn [option_map]. f_equal.
    exact (bind_table_load_renamed r v s first _ _ (Apply_cont merged)).
  - intros [s elements] Hin. apply table_has_arms_renamed.
    exact (Hclosed s elements Hin).
Qed.

(* A way for each argument, renamed: a table is as it was, its
   elements having no variable. *)
Definition rename_table_way (r : renaming) (way : table_way) : table_way :=
  match way with
  | Without_table way => Without_table (rename_way r way)
  | From_table s elements immediates => From_table s elements immediates
  end.

Definition table_way_closed (way : table_way) : bool :=
  match way with
  | Without_table _ => true
  | From_table _ elements _ => forallb element_closed elements
  end.

Lemma argument_by_table_way_renamed (r : renaming) (first offset i : Z)
    (way : table_way) :
  table_way_closed way = true ->
  argument_by_table_way first offset i (rename_table_way r way)
  = option_map (rename_simple r) (argument_by_table_way first offset i way).
Proof.
  intros Hclosed.
  destruct way as [way | s elements immediates];
    cbn [rename_table_way argument_by_table_way];
    [ apply argument_in_arm_renamed | ].
  destruct (table_element first offset elements i) as [element | ]
    eqn:Helement; [ | reflexivity ].
  cbn beta iota.
  destruct (element_taken _ element); [ | reflexivity ].
  cbn [option_map].
  rewrite
    (closed_element_renamed r element
       (table_element_closed first offset elements i element Hclosed
          Helement)).
  reflexivity.
Qed.

Lemma arguments_by_table_ways_renamed (r : renaming) (first offset i : Z)
    (ways : list table_way) :
  forallb table_way_closed ways = true ->
  arguments_by_table_ways first offset i (map (rename_table_way r) ways)
  = option_map (map (rename_simple r))
      (arguments_by_table_ways first offset i ways).
Proof.
  induction ways as [ | way ways IH]; [ reflexivity | ].
  cbn [forallb]. intros Hclosed.
  apply andb_prop in Hclosed as [Hway Hclosed].
  cbn [map arguments_by_table_ways].
  rewrite (argument_by_table_way_renamed r first offset i way Hway),
    (IH Hclosed).
  destruct (argument_by_table_way first offset i way); [ | reflexivity ].
  destruct (arguments_by_table_ways first offset i ways); reflexivity.
Qed.

Lemma arm_has_table_ways_renamed (r : renaming) (k : continuation)
    (first offset : Z) (ways : list table_way) (arm : Z * apply_cont) :
  forallb table_way_closed ways = true ->
  arm_has_table_ways (rename_continuation r k) first offset
    (map (rename_table_way r) ways) (fst arm, rename_apply_cont r (snd arm))
  = arm_has_table_ways k first offset ways arm.
Proof.
  intros Hclosed. unfold arm_has_table_ways. cbn [fst snd].
  rewrite
    (arguments_by_table_ways_renamed r first offset (fst arm) ways Hclosed).
  destruct (arguments_by_table_ways first offset (fst arm) ways)
    as [args | ]; [ | reflexivity ].
  cbn [option_map].
  change
    (Apply_cont_expr (rename_continuation r k) (map (rename_simple r) args)
       None)
    with (rename_apply_cont r (Apply_cont_expr k args None)).
  destruct (apply_cont_eq_dec (snd arm) (Apply_cont_expr k args None))
    as [-> | Hne],
    (apply_cont_eq_dec (rename_apply_cont r _) (rename_apply_cont r _))
      as [Heq | Hne'];
    try reflexivity; [ contradiction | ].
  elim Hne. exact (renamed_apply_cont_same r _ _ Heq).
Qed.

Lemma table_has_column_renamed (r : renaming) (first offset : Z)
    (column : list (Z * simple)) (table : symbol * list simple) :
  forallb element_closed (snd table) = true ->
  table_has_column first offset (map (rename_entry r) column) table
  = table_has_column first offset column table.
Proof.
  intros Hclosed. unfold table_has_column. rewrite length_map. f_equal.
  cbv zeta. apply forallb_under_map. intros entry.
  cbn [rename_entry fst snd].
  destruct (table_element first offset (snd table) (fst entry))
    as [element | ] eqn:Helement; [ | reflexivity ].
  f_equal.
  pose proof (simple_is_renamed r (snd entry) element) as Hsame.
  rewrite
    (closed_element_renamed r element
       (table_element_closed first offset _ _ element Hclosed Helement))
    in Hsame.
  exact Hsame.
Qed.

Lemma table_way_of_column_renamed (r : renaming) (tables : known_tables)
    (first offset : Z) (column : list (Z * simple)) :
  tables_closed tables ->
  table_way_of_column tables first offset (map (rename_entry r) column)
  = option_map (rename_table_way r)
      (table_way_of_column tables first offset column).
Proof.
  intros Hclosed. unfold table_way_of_column.
  rewrite way_of_column_renamed.
  destruct (way_of_column column) as [way | ]; [ reflexivity | ].
  cbn [option_map].
  rewrite (find_same_among _ (table_has_column first offset column) tables).
  - destruct (find _ tables) as [[s elements] | ]; reflexivity.
  - intros [s elements] Hin. apply table_has_column_renamed.
    exact (Hclosed s elements Hin).
Qed.

Lemma table_ways_of_arguments_renamed (r : renaming) (tables : known_tables)
    (first offset : Z) :
  tables_closed tables ->
  ∀ (count : nat) (rows : list (Z * list simple)),
    table_ways_of_arguments tables first offset count
      (map (rename_row r) rows)
    = option_map (map (rename_table_way r))
        (table_ways_of_arguments tables first offset count rows).
Proof.
  intros Hclosed.
  induction count as [ | count IH]; intros rows; [ reflexivity | ].
  cbn [table_ways_of_arguments]. rewrite column_of_arms_renamed.
  destruct (column_of_arms rows) as [column | ]; [ | reflexivity ].
  cbn [option_map].
  rewrite (table_way_of_column_renamed r tables first offset column Hclosed).
  destruct (table_way_of_column tables first offset column) as [way | ];
    [ | reflexivity ].
  cbn [option_map]. rewrite rest_of_arms_renamed, IH.
  destruct
    (table_ways_of_arguments tables first offset count (rest_of_arms rows));
    reflexivity.
Qed.

Lemma table_ways_of_arguments_closed (tables : known_tables)
    (first offset : Z) :
  tables_closed tables ->
  ∀ (count : nat) (rows : list (Z * list simple)) (ways : list table_way),
    table_ways_of_arguments tables first offset count rows = Some ways ->
    forallb table_way_closed ways = true.
Proof.
  intros Hclosed.
  induction count as [ | count IH]; intros rows ways Hways;
    cbn [table_ways_of_arguments] in Hways.
  - injection Hways as <-. reflexivity.
  - destruct (column_of_arms rows) as [column | ]; [ | discriminate ].
    cbn beta iota in Hways.
    destruct (table_way_of_column tables first offset column) as [way | ]
      eqn:Hway; [ | discriminate ].
    cbn beta iota in Hways.
    destruct
      (table_ways_of_arguments tables first offset count (rest_of_arms rows))
      as [ways0 | ] eqn:Hways0; [ | discriminate ].
    injection Hways as <-.
    cbn [forallb]. rewrite (IH _ _ Hways0), andb_true_r.
    unfold table_way_of_column in Hway.
    destruct (way_of_column column); [ injection Hway as <-; reflexivity | ].
    destruct (find _ tables) as [[s elements] | ] eqn:Hfind;
      [ | discriminate ].
    injection Hway as <-.
    apply find_some in Hfind as [Hin _]. exact (Hclosed s elements Hin).
Qed.

Lemma bind_table_load_to_renamed (r : renaming) (v x : variable) (s : symbol)
    (first offset : Z) (immediates : bool) (body : expr) :
  rename_expr r (bind_table_load_to v x s first offset immediates body)
  = bind_table_load_to (rename_variable r v) (rename_variable r x) s first
      offset immediates (rename_expr r body).
Proof.
  unfold bind_table_load_to, bind_tagged_to, bind_table_offset.
  destruct (first =? 0); reflexivity.
Qed.

Lemma bind_table_ways_renamed (r : renaming) (first offset : Z)
    (v : variable) (body : expr) (names : list variable) :
  ∀ ways : list table_way,
    rename_expr r (bind_table_ways first offset v names ways body)
    = bind_table_ways first offset (rename_variable r v)
        (map (rename_variable r) names) (map (rename_table_way r) ways)
        (rename_expr r body).
Proof.
  induction names as [ | x names IH];
    intros [ | [[s | how] | s elements immediates] ways];
    cbn [bind_table_ways bind_table_way map rename_table_way rename_way];
    try reflexivity.
  - apply IH.
  - rewrite bind_way_renamed, IH. reflexivity.
  - rewrite bind_table_load_to_renamed, IH. reflexivity.
Qed.

Lemma merged_by_table_ways_renamed (r : renaming) (v : variable)
    (names : list variable) :
  ∀ ways : list table_way,
    map (rename_simple r) (merged_by_table_ways v names ways)
    = merged_by_table_ways (rename_variable r v)
        (map (rename_variable r) names) (map (rename_table_way r) ways).
Proof.
  induction names as [ | x names IH];
    intros [ | [[s | how] | s elements immediates] ways];
    cbn [merged_by_table_ways merged_by_table_way map rename_table_way
      rename_way];
    try reflexivity; rewrite IH; try reflexivity.
  destruct how; reflexivity.
Qed.

Lemma table_names_usable_renamed (r : renaming) (v : variable)
    (names : list variable) (ways : list table_way) :
  table_names_usable (rename_variable r v) (map (rename_variable r) names)
    (map (rename_table_way r) ways)
  = table_names_usable v names ways.
Proof.
  unfold table_names_usable.
  rewrite !length_map, distinct_names_renamed, among_names_renamed.
  f_equal. apply forallb_under_map.
  intros [[s | how] | s elements immediates]; try reflexivity.
  cbn [rename_table_way rename_way]. rewrite names_in_simple_renamed.
  reflexivity.
Qed.

Lemma is_from_table_renamed (r : renaming) (ways : list table_way) :
  existsb is_from_table (map (rename_table_way r) ways)
  = existsb is_from_table ways.
Proof.
  induction ways as [ | [way | s elements immediates] ways IH];
    [ reflexivity | | ];
    cbn [map existsb rename_table_way is_from_table]; rewrite ?IH;
    reflexivity.
Qed.

Lemma merge_switch_by_table_ways_renamed (r : renaming)
    (known : known_continuations) (tables : known_tables)
    (scrutinee : simple) (arms : list (Z * apply_cont)) :
  tables_closed tables ->
  merge_switch_by_table_ways (rename_known_continuations r known) tables
    (rename_simple r scrutinee)
    (map (fun arm => (fst arm, rename_apply_cont r (snd arm))) arms)
  = option_map (rename_expr r)
      (merge_switch_by_table_ways known tables scrutinee arms).
Proof.
  intros Hclosed.
  destruct scrutinee as [v | sym | c]; try reflexivity.
  destruct arms as [ | [first ac] [ | arm1 [ | arm2 others]]];
    try reflexivity.
  set (arms := (first, ac) :: arm1 :: arm2 :: others).
  set (offset := table_offset first).
  assert
    (Hrows :
      map (fun arm => (fst arm, ac_args (snd arm)))
        (map (fun arm => (fst arm, rename_apply_cont r (snd arm))) arms)
      = map (rename_row r)
          (map (fun arm => (fst arm, ac_args (snd arm))) arms)).
  { rewrite !map_map. apply map_ext. intros arm. reflexivity. }
  change
    (merge_switch_by_table_ways (rename_known_continuations r known) tables
       (rename_simple r (Var v))
       (map (fun arm => (fst arm, rename_apply_cont r (snd arm))) arms))
    with
      (let* names :=
         lookup (rename_continuation r (ac_continuation ac))
           (rename_known_continuations r known)
       in
       let* ways :=
         table_ways_of_arguments tables first offset
           (length (map (rename_simple r) (ac_args ac)))
           (map (fun arm => (fst arm, ac_args (snd arm)))
              (map (fun arm => (fst arm, rename_apply_cont r (snd arm)))
                 arms))
       in
       if existsb is_from_table ways
          && table_names_usable (rename_variable r v) names ways
          && forallb
               (arm_has_table_ways
                  (rename_continuation r (ac_continuation ac)) first offset
                  ways)
               (map (fun arm => (fst arm, rename_apply_cont r (snd arm)))
                  arms)
       then
         Some
           (bind_table_ways first offset (rename_variable r v) names ways
              (Apply_cont
                 (Apply_cont_expr (rename_continuation r (ac_continuation ac))
                    (merged_by_table_ways (rename_variable r v) names ways)
                    None)))
       else None).
  change (merge_switch_by_table_ways known tables (Var v) arms)
    with
      (let* names := lookup (ac_continuation ac) known in
       let* ways :=
         table_ways_of_arguments tables first offset (length (ac_args ac))
           (map (fun arm => (fst arm, ac_args (snd arm))) arms)
       in
       if existsb is_from_table ways
          && table_names_usable v names ways
          && forallb (arm_has_table_ways (ac_continuation ac) first offset ways)
               arms
       then
         Some
           (bind_table_ways first offset v names ways
              (Apply_cont
                 (Apply_cont_expr (ac_continuation ac)
                    (merged_by_table_ways v names ways) None)))
       else None).
  rewrite lookup_known_renamed.
  destruct (lookup (ac_continuation ac) known) as [names | ];
    [ | reflexivity ].
  cbn [option_map].
  rewrite Hrows, length_map,
    (table_ways_of_arguments_renamed r tables first offset Hclosed).
  destruct (table_ways_of_arguments _ _ _ _ _) as [ways | ] eqn:Hways;
    [ | reflexivity ].
  cbn [option_map].
  pose proof
    (table_ways_of_arguments_closed tables first offset Hclosed _ _ ways
       Hways)
    as Hways_closed.
  rewrite is_from_table_renamed, table_names_usable_renamed,
    (forallb_under_map _ _
       (arm_has_table_ways (ac_continuation ac) first offset ways))
    by (intros arm; apply arm_has_table_ways_renamed, Hways_closed).
  destruct (_ && _ && _); [ | reflexivity ].
  cbn [option_map]. rewrite bind_table_ways_renamed.
  cbn [rename_expr]. unfold rename_apply_cont.
  cbn [ac_continuation ac_args ac_trap_action option_map].
  rewrite merged_by_table_ways_renamed. reflexivity.
Qed.

Lemma merge_switch_with_tables_renamed (r : renaming)
    (known : known_continuations) (tables : known_tables)
    (scrutinee : simple) (arms : list (Z * apply_cont)) :
  tables_closed tables ->
  merge_switch_with_tables (rename_known_continuations r known) tables
    (rename_simple r scrutinee)
    (map (fun arm => (fst arm, rename_apply_cont r (snd arm))) arms)
  = rename_expr r (merge_switch_with_tables known tables scrutinee arms).
Proof.
  intros Hclosed. rewrite !merge_switch_with_tables_unfold.
  rewrite merge_switch_equivariant,
    (merge_switch_by_table_renamed r tables scrutinee arms Hclosed),
    (merge_switch_by_table_ways_renamed r known tables scrutinee arms
       Hclosed).
  destruct (merge_switch_by_table tables scrutinee arms); cbn [option_map].
  - destruct (merge_switch scrutinee arms); reflexivity.
  - destruct (merge_switch_by_table_ways known tables scrutinee arms);
      reflexivity.
Qed.

Fixpoint merge_tables_under_equivariant (r : renaming) (e : expr) :
  ∀ (known : known_continuations) (tables : known_tables),
    tables_closed tables ->
    merge_tables_under (rename_known_continuations r known) tables
      (rename_expr r e)
    = rename_expr r (merge_tables_under known tables e)
with merge_tables_in_handler_equivariant (r : renaming)
    (h : continuation_handler) :
  ∀ (known : known_continuations) (tables : known_tables),
    tables_closed tables ->
    merge_tables_in_handler (rename_known_continuations r known) tables
      (rename_handler r h)
    = rename_handler r (merge_tables_in_handler known tables h).
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      intros known tables Hclosed; cbn [rename_expr merge_tables_under];
      try reflexivity.
    + rewrite (merge_tables_under_equivariant r body known tables Hclosed).
      reflexivity.
    + rewrite tables_under_statics_renamed,
        (merge_tables_under_equivariant r body known _
           (tables_under_statics_closed tables bindings Hclosed)).
      reflexivity.
    + destruct h as [params metadata handler].
      cbn [rename_handler merge_tables_in_handler parameters_of_handler].
      change
        ((rename_continuation r k, map (rename_variable r) params)
         :: rename_known_continuations r known)
        with (rename_known_continuations r ((k, params) :: known)).
      rewrite (merge_tables_under_equivariant r handler known tables Hclosed),
        (merge_tables_under_equivariant r body _ tables Hclosed).
      reflexivity.
    + cbv zeta.
      rewrite known_of_handlers_renamed,
        (merge_tables_under_equivariant r body _ tables Hclosed).
      f_equal.
      generalize
        (map (fun entry => (fst entry, parameters_of_handler (snd entry)))
           handlers
         ++ known).
      intros known'.
      induction handlers as [ | [k h] handlers IH]; simpl; [ reflexivity | ].
      rewrite (merge_tables_in_handler_equivariant r h known' tables Hclosed),
        IH.
      reflexivity.
    + apply merge_switch_with_tables_renamed, Hclosed.
  - destruct h as [params metadata handler]. intros known tables Hclosed.
    cbn [rename_handler merge_tables_in_handler].
    rewrite (merge_tables_under_equivariant r handler known tables Hclosed).
    reflexivity.
Qed.

Lemma merge_tables_knowing_equivariant (r : renaming) (e : expr)
    (tables : known_tables) :
  tables_closed tables ->
  merge_tables_knowing tables (rename_expr r e)
  = rename_expr r (merge_tables_knowing tables e).
Proof. exact (merge_tables_under_equivariant r e [] tables). Qed.

Lemma merged_body_renamed (tables : known_tables) (r : renaming)
    (body : expr) :
  tables_closed tables ->
  rewrite_code_body (merge_tables_knowing tables) (rename_expr r body)
  = rename_expr r (rewrite_code_body (merge_tables_knowing tables) body).
Proof.
  intros Hclosed. unfold rewrite_code_body.
  rewrite (merge_tables_knowing_equivariant r body tables Hclosed),
    !rename_binds_statics.
  destruct (Bool.eqb (binds_statics body)
              (binds_statics (merge_tables_knowing tables body)));
    reflexivity.
Qed.

Lemma merged_binding_renamed (tables : known_tables) (r : renaming)
    (binding : static_binding) :
  tables_closed tables ->
  rewrite_code_in_binding (merge_tables_knowing tables)
    (rename_static_binding r binding)
  = rename_static_binding r
      (rewrite_code_in_binding (merge_tables_knowing tables) binding).
Proof.
  intros Hclosed.
  destruct binding
    as [id [params my_closure ret exn metadata body] | id | sym const];
    try reflexivity.
  cbn [rename_static_binding rename_code rewrite_code_in_binding].
  rewrite (merged_body_renamed tables _ _ Hclosed). reflexivity.
Qed.

Fixpoint merge_tables_in_code_equivariant (r : renaming) (e : expr) :
  ∀ tables : known_tables,
    tables_closed tables ->
    merge_tables_in_code tables (rename_expr r e)
    = rename_expr r (merge_tables_in_code tables e)
with merge_tables_in_code_in_handler_equivariant (r : renaming)
    (h : continuation_handler) :
  ∀ tables : known_tables,
    tables_closed tables ->
    merge_tables_in_code_in_handler tables (rename_handler r h)
    = rename_handler r (merge_tables_in_code_in_handler tables h).
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      intros tables Hclosed; cbn [rename_expr merge_tables_in_code];
      cbv zeta; try reflexivity.
    + rewrite (merge_tables_in_code_equivariant r body tables Hclosed).
      reflexivity.
    + rewrite tables_under_statics_renamed,
        (merge_tables_in_code_equivariant r body _
           (tables_under_statics_closed tables bindings Hclosed)),
        !map_map.
      f_equal. apply map_ext. intros binding.
      apply merged_binding_renamed, tables_under_statics_closed, Hclosed.
    + rewrite
        (merge_tables_in_code_in_handler_equivariant r h tables Hclosed),
        (merge_tables_in_code_equivariant r body tables Hclosed).
      reflexivity.
    + rewrite (merge_tables_in_code_equivariant r body tables Hclosed).
      f_equal.
      induction handlers as [ | [k h] handlers IH]; simpl; [ reflexivity | ].
      rewrite
        (merge_tables_in_code_in_handler_equivariant r h tables Hclosed), IH.
      reflexivity.
  - destruct h as [params metadata handler]. intros tables Hclosed.
    cbn [rename_handler merge_tables_in_code_in_handler].
    rewrite (merge_tables_in_code_equivariant r handler tables Hclosed).
    reflexivity.
Qed.

Theorem merge_tables_equivariant : merge_tables_equivariant_statement.
Proof.
  intros r e. unfold merge_tables.
  rewrite (merge_tables_knowing_equivariant r e [] no_tables_closed).
  exact (merge_tables_in_code_equivariant r _ [] no_tables_closed).
Qed.

(* ======================================================================== *)
(* A term with no table                                                     *)
(* ======================================================================== *)

Lemma no_tables_under_statics (bindings : list static_binding) :
  forallb binding_binds_no_table bindings = true ->
  tables_under_statics [] bindings = [].
Proof.
  unfold tables_under_statics. rewrite app_nil_r.
  induction bindings as [ | binding bindings IH]; [ reflexivity | ].
  cbn [forallb flat_map]. intros Hnone.
  apply andb_prop in Hnone as [Hbinding Hnone]. rewrite (IH Hnone).
  destruct binding as [id c | id | s const]; try reflexivity.
  destruct const; try reflexivity. discriminate Hbinding.
Qed.

Fixpoint merge_tables_under_without_tables (e : expr) :
  ∀ known : known_continuations,
    binds_no_table e = true -> merge_tables_under known [] e = e
with merge_tables_in_handler_without_tables (h : continuation_handler) :
  ∀ known : known_continuations,
    handler_binds_no_table h = true ->
    merge_tables_in_handler known [] h = h.
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      intros known; cbn [binds_no_table merge_tables_under]; intros Hnone;
      try reflexivity.
    + rewrite (merge_tables_under_without_tables body known Hnone).
      reflexivity.
    + apply andb_prop in Hnone as [Hbindings Hbody].
      rewrite (no_tables_under_statics bindings Hbindings),
        (merge_tables_under_without_tables body known Hbody).
      reflexivity.
    + apply andb_prop in Hnone as [Hhandler Hbody].
      rewrite (merge_tables_in_handler_without_tables h known Hhandler),
        (merge_tables_under_without_tables body _ Hbody).
      reflexivity.
    + cbv zeta. apply andb_prop in Hnone as [Hhandlers Hbody].
      rewrite (merge_tables_under_without_tables body _ Hbody). f_equal.
      generalize
        (map (fun entry => (fst entry, parameters_of_handler (snd entry)))
           handlers
         ++ known).
      intros known'.
      induction handlers as [ | [k h] handlers IH]; [ reflexivity | ].
      cbn [forallb map] in Hhandlers |- *.
      apply andb_prop in Hhandlers as [Hhandler Hhandlers].
      rewrite (merge_tables_in_handler_without_tables h known' Hhandler),
        (IH Hhandlers).
      reflexivity.
  - destruct h as [params metadata handler]. intros known.
    cbn [handler_binds_no_table merge_tables_in_handler]. intros Hnone.
    rewrite (merge_tables_under_without_tables handler known Hnone).
    reflexivity.
Qed.

Lemma merge_tables_knowing_without_tables (e : expr) :
  binds_no_table e = true -> merge_tables_knowing [] e = e.
Proof. apply merge_tables_under_without_tables. Qed.

Lemma merged_binding_without_tables (binding : static_binding) :
  binding_binds_no_table binding = true ->
  rewrite_code_in_binding (merge_tables_knowing []) binding = binding.
Proof.
  destruct binding
    as [id [params my_closure ret exn metadata body] | id | s const];
    try reflexivity.
  cbn [binding_binds_no_table code_binds_no_table rewrite_code_in_binding].
  intros Hnone. unfold rewrite_code_body.
  rewrite (merge_tables_knowing_without_tables body Hnone).
  destruct (Bool.eqb _ _); reflexivity.
Qed.

Fixpoint merge_tables_in_code_without_tables (e : expr) :
  binds_no_table e = true -> merge_tables_in_code [] e = e
with merge_tables_in_code_in_handler_without_tables
    (h : continuation_handler) :
  handler_binds_no_table h = true ->
  merge_tables_in_code_in_handler [] h = h.
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      cbn [binds_no_table merge_tables_in_code]; cbv zeta; intros Hnone;
      try reflexivity.
    + rewrite (merge_tables_in_code_without_tables body Hnone). reflexivity.
    + apply andb_prop in Hnone as [Hbindings Hbody].
      rewrite (no_tables_under_statics bindings Hbindings),
        (merge_tables_in_code_without_tables body Hbody).
      f_equal.
      induction bindings as [ | binding bindings IH]; [ reflexivity | ].
      cbn [forallb map] in Hbindings |- *.
      apply andb_prop in Hbindings as [Hbinding Hbindings].
      rewrite (merged_binding_without_tables binding Hbinding),
        (IH Hbindings).
      reflexivity.
    + apply andb_prop in Hnone as [Hhandler Hbody].
      rewrite (merge_tables_in_code_in_handler_without_tables h Hhandler),
        (merge_tables_in_code_without_tables body Hbody).
      reflexivity.
    + apply andb_prop in Hnone as [Hhandlers Hbody].
      rewrite (merge_tables_in_code_without_tables body Hbody). f_equal.
      induction handlers as [ | [k h] handlers IH]; [ reflexivity | ].
      cbn [forallb map] in Hhandlers |- *.
      apply andb_prop in Hhandlers as [Hhandler Hhandlers].
      rewrite (merge_tables_in_code_in_handler_without_tables h Hhandler),
        (IH Hhandlers).
      reflexivity.
  - destruct h as [params metadata handler].
    cbn [handler_binds_no_table merge_tables_in_code_in_handler].
    intros Hnone.
    rewrite (merge_tables_in_code_without_tables handler Hnone). reflexivity.
Qed.

Theorem merge_tables_without_tables : merge_tables_without_tables_statement.
Proof.
  intros e Hnone. unfold merge_tables.
  rewrite (merge_tables_knowing_without_tables e Hnone).
  exact (merge_tables_in_code_without_tables e Hnone).
Qed.

(* ======================================================================== *)
(* Four terms                                                               *)
(* ======================================================================== *)

Theorem merge_tables_example : merge_tables_example_statement.
Proof. vm_compute. reflexivity. Qed.

Theorem merge_tables_equal_elements_example :
  merge_tables_equal_elements_example_statement.
Proof. vm_compute. reflexivity. Qed.

Theorem merge_tables_two_tables_example :
  merge_tables_two_tables_example_statement.
Proof. vm_compute. reflexivity. Qed.

Theorem merge_tables_table_and_line_example :
  merge_tables_table_and_line_example_statement.
Proof. vm_compute. reflexivity. Qed.
