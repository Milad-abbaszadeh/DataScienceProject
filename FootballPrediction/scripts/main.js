/* Semantic UI Elements Initializations */
$('select.dropdown')
    .dropdown()
;

$('.ui.dropdown')
    .dropdown()
;




$(document).ready(function () {
    $('#doughnut-chart').hide();
    $('.dropdown').change(function () {
        if($('#team1').val() == ""){
            $('#flag1').attr("src","images/question.png");
        }
        else{
            $('#flag1').attr("src","images/"+$('#team1').val()+".png");
            
        }
        if($('#team2').val() == ""){
            $('#flag2').attr("src","images/question.png");
        }
        else{
            $('#flag2').attr("src","images/"+$('#team2').val()+".png");
        }
    });
});

$('#predict').click(function () {
    if( $('#team1').val() != "" && $('#team2').val() != "" && $('#team1').val() != $('#team2').val()){
        $('#error-message').hide();
        $('#doughnut-chart').show();
        new Chart( $('#doughnut-chart'), {
            type: 'doughnut',
            data: {
                labels: [$('#team1').val()+" Win", "EQUAL", $('#team2').val()+" Win"],
                datasets: [
                    {
                        backgroundColor: ["#19ff1a", "yellow","orange"],
                        data: [60,15,25]
                    }
                ]
            },
            options: {
                title: {
                    display: true,
                    fontSize: 18,
                    fontColor: "#fff",
                    fontFamily: "dusha_plusregular,'dusha_plusregular', tahoma",
                    text: 'The chance of winning prediction ('+$('#team1').val()+' vs. '+$('#team2').val()+')'
                }
            }
        });
    }
    else{
        $('#error-message').text('Selected Teams are invalid! Please try again...');
        $('#error-message').show();
        $('#doughnut-chart').hide();
    }

}
);